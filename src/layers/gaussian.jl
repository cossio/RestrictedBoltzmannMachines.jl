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

function conjugates(layer::Gaussian)
    θ = layer.θ
    γ = abs.(layer.γ)
    return (
        θ = θ ./ γ,
        γ = @. -sign(γ) * (γ + θ^2) / γ^2 / 2
    )
end

function conjugates_empirical(layer::Gaussian, samples::AbstractArray)
    @assert size(samples) == (size(layer)..., size(samples)[end])
    μ = mean_(samples; dims=ndims(samples))
    μ2 = mean_(samples.^2; dims=ndims(samples))
    return (θ = μ, γ = -μ2/2)
end

function effective(layer::Gaussian, inputs, β::Real = true)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return Gaussian(promote(θ, γ)...)
end

gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x
gauss_cgf(θ::Real, γ::Real) = θ^2 / abs(2γ) - log(abs(γ)/π/2) / 2
gauss_mode(θ::Real, γ::Real) = θ / abs(γ)
