"""
    Gaussian(θ, γ)

Gaussian layer, with location parameters `θ` and scale parameters `γ`.
"""
struct Gaussian{A<:AbstractArray}
    θ::A
    γ::A
    function Gaussian(θ::A, γ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ)
        return new{A}(θ, γ)
    end
end

function Gaussian(::Type{T}, n::Int...) where {T}
    return Gaussian(zeros(T, n...), ones(T, n...))
end

Gaussian(n::Int...) = Gaussian(Float64, n...)

Flux.@functor Gaussian

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)
cgfs(layer::Gaussian) = gauss_cgf.(layer.θ, layer.γ)

function sample(layer::Gaussian)
    μ = @. layer.θ / abs(layer.γ)
    σ = @. inv(sqrt(abs(layer.γ)))
    z = randn(eltype(μ), size(μ))
    return μ .+ σ .* z
end

mode(layer::Gaussian) = gauss_mode.(layer.θ, layer.γ)

function effective(layer::Gaussian, inputs, β::Real = 1)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return Gaussian(promote(θ, γ)...)
end

function gauss_energy(θ::Real, γ::Real, x::Real)
    return (abs(γ) * x / 2 - θ) * x
end

function gauss_cgf(θ::Real, γ::Real)
    return θ^2 / abs(2γ) - log(abs(γ)/π/2) / 2
end

gauss_mode(θ::Real, γ::Real) = θ / abs(γ)
