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
    E = drelu_energy.(layer.θp, layer.θn, layer.γp, layer.γn, x)
    return sum_(E; dims = layerdims(layer))
end

function cgf(layer::dReLU, inputs::AbstractArray)
    Γ = drelu_cgf.(inputs .+ layer.θp, inputs .+ layer.θn, layer.γp, layer.γn)
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

function drelu_energy(θp::Real, θn::Real, γp::Real, γn::Real, x::Real)
    xp = max(x, zero(x))
    xn = min(x, zero(x))
    Ep = (abs(γp) * xp / 2 - θp) * xp
    En = (abs(γn) * xn / 2 - θn) * xn
    return Ep + En
end

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf( θp, γp)
    Γn = relu_cgf(-θn, γn)
    return LogExpFunctions.logaddexp(Γp, Γn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf( θp, γp)
    Γn = relu_cgf(-θn, γn)
    Γ = LogExpFunctions.logaddexp(Γp, Γn)
    if randexp(typeof(Γ)) ≥ Γ - Γp
        return  relu_rand( θp, γp)
    else
        return -relu_rand(-θn, γn)
    end
end

Base.size(layer::dReLU) = size(layer.θp)
Base.size(layer::dReLU, d::Int) = size(layer.θp, d)
Base.ndims(layer::dReLU) = ndims(layer.θp)
Base.length(layer::dReLU) = length(layer.θp)
