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

energies(layer::ReLU, x) = relu_energy.(layer.θ, layer.γ, x)
cgfs(layer::ReLU) = relu_cgf.(layer.θ, layer.γ)
sample(layer::ReLU) = relu_rand.(layer.θ, layer.γ)

function effective(layer::ReLU, inputs, β::Real = 1)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return ReLU(promote(θ, γ)...)
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
    abs_γ = abs(γ)
    return SpecialFunctions.logerfcx(-θ / √(2abs_γ)) - log(2abs_γ/π) / 2
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end
