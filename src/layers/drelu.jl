struct dReLU{A<:AbstractArray}
    θp::A
    θn::A
    γp::A
    γn::A
    function dReLU(θp::A, θn::A, γp::A, γn::A) where {A<:AbstractArray}
        @assert size(θp) == size(θn) == size(γp) == size(γn)
        return new{A}(θp, θn, γp, γn)
    end
end

function dReLU(::Type{T}, n::Int...) where {T}
    θp = zeros(T, n...)
    θn = zeros(T, n...)
    γp = ones(T, n...)
    γn = ones(T, n...)
    return dReLU(θp, θn, γp, γn)
end

dReLU(n::Int...) = dReLU(Float64, n...)

Flux.@functor dReLU

function energy(layer::dReLU, x::AbstractArray)
    xp = @. max( x, zero(x))
    xn = @. max(-x, zero(x))
    Ep = energy(ReLU( layer.θp, layer.γp), xp)
    En = energy(ReLU(-layer.θn, layer.γn), xn)
    return Ep .+ En
end

function cgf(layer::dReLU, inputs::AbstractArray)
    Γp = relu_cgf.(inputs .+ layer.θp, layer.γp)
    Γn = relu_cgf.(inputs .- layer.θn, layer.γn)
    Γ = logaddexp.(Γp, Γn)
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::dReLU, inputs::AbstractArray, β::Real)
    layer_ = dReLU(layer.θp .* β, layer.θn .* β, layer.γp .* β, layer.γn .* β)
    return cgf(layer_, inputs .* β) / β
end

function sample_from_inputs(layer::dReLU, inputs::AbstractArray)
    return @. drelu_rand(layer.θp + inputs, layer.θn + inputs, layer.γp, layer.γn)
end

function sample_from_inputs(layer::dReLU, inputs::AbstractArray, β::Real)
    layer_ = dReLU(β .* layer.θp, β .* layer.θn, β .* layer.γp, β .* layer.γn)
    return sample_from_inputs(layer_, inputs .* β)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf(+θp, γp)
    Γn = relu_cgf(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    if rand(typeof(Γ)) ≤ exp(Γp - Γ)
        return +relu_rand(+θp, γp)
    else
        return -relu_rand(-θn, γn)
    end
end

Base.size(layer::dReLU) = size(layer.θp)
Base.ndims(layer::dReLU) = ndims(layer.θp)
Base.length(layer::dReLU) = length(layer.θp)
