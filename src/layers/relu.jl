"""
    ReLU(θ, γ)

ReLU layer, with location parameters `θ` and scale parameters `γ`.
"""
struct ReLU{A<:AbstractArray}
    θ::A
    γ::A
    function ReLU(θ::A, γ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ)
        return new{A}(θ, γ)
    end
end

ReLU(::Type{T}, n::Int...) where {T} = ReLU(zeros(T, n...), ones(T, n...))
ReLU(n::Int...) = ReLU(Float64, n...)

Flux.@functor ReLU

function energy(layer::ReLU, x::AbstractArray)
    @assert size(x) == (size(layer)..., size(x)[end])
    E = relu_energy.(layer.θ, layer.γ, x)
    return sum_(E; dims = layerdims(layer))
end

function cgf(layer::ReLU, inputs::AbstractArray)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    Γ = relu_cgf.(layer.θ .+ inputs, layer.γ)
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::ReLU, inputs::AbstractArray, β::Real)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    layer_ = ReLU(layer.θ .* β, layer.γ .* β)
    return cgf(layer_, inputs .* β) / β
end

function sample_from_inputs(layer::ReLU, inputs::AbstractArray)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    return relu_rand.(layer.θ .+ inputs, layer.γ)
end

function sample_from_inputs(layer::ReLU, inputs::AbstractArray, β::Real)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    layer_ = ReLU(layer.θ .* β, layer.γ .* β)
    return sample_from_inputs(layer_, inputs .* β)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    if x < 0
        return inf(E)
    else
        return E
    end
end

function relu_cgf(θ::Real, γ::Real)
    γa = abs(γ)
    return SpecialFunctions.logerfcx(-θ / √(2γa)) - log(2γa/π) / 2
end

function relu_rand(θ::Real, γ::Real)
    γa = abs(γ)
    μ = θ / γa
    σ = √inv(γa)
    return randnt_half(μ, σ)
end
