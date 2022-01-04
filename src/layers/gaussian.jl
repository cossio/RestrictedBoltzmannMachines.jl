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

Gaussian(::Type{T}, n::Int...) where {T} = Gaussian(zeros(T, n...), ones(T, n...))
Gaussian(n::Int...) = Gaussian(Float64, n...)

Flux.@functor Gaussian

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)
free_energies(layer::Gaussian) = gauss_free.(layer.θ, layer.γ)

function transfer_sample(layer::Gaussian)
    μ = transfer_mean(layer)
    σ = sqrt.(transfer_var(layer))
    z = randn(eltype(μ), size(μ))
    return μ .+ σ .* z
end

transfer_mode(layer::Gaussian) = gauss_mode.(layer.θ, layer.γ)
transfer_mean(layer::Gaussian) = transfer_mode(layer)
transfer_var(layer::Gaussian) = inv.(abs.(layer.γ))

function transfer_mean_abs(layer::Gaussian)
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    return @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * SpecialFunctions.erf(μ / √(2ν))
end

function effective(layer::Gaussian, inputs; β::Real = true)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return Gaussian(promote(θ, γ)...)
end

function ∂free_energy(layer::Gaussian)
    θ = layer.θ
    γ = abs.(layer.γ)
    return (
        θ = (@. -θ / γ),
        γ = (@. sign(γ) * (γ + θ^2) / γ^2 / 2)
    )
end

∂energies(::Gaussian, x::AbstractArray) = (θ = -x, γ = @. x^2 / 2)

gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x
gauss_free(θ::Real, γ::Real) = -θ^2 / abs(2γ) + log(abs(γ)/π/2) / 2
gauss_mode(θ::Real, γ::Real) = θ / abs(γ)
