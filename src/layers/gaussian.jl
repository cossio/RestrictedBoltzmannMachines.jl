"""
    Gaussian(θ, γ)

Gaussian layer, with location parameters `θ` and scale parameters `γ`.
"""
struct Gaussian{N,Aθ,Aγ} <: AbstractLayer{N}
    θ::Aθ
    γ::Aγ
    function Gaussian(θ::AbstractArray, γ::AbstractArray)
        @assert size(θ) == size(γ)
        return new{ndims(θ), typeof(θ), typeof(γ)}(θ, γ)
    end
end

Gaussian(::Type{T}, n::Int...) where {T} = Gaussian(zeros(T, n...), ones(T, n...))
Gaussian(n::Int...) = Gaussian(Float64, n...)
StdGauss(n::Int...) = Gaussian(FillArrays.Falses(n), FillArrays.Trues(n))

function effective(layer::Gaussian, inputs::AbstractArray; β::Real = true)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return Gaussian(θ, γ)
end

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)
free_energies(layer::Gaussian) = gauss_free.(layer.θ, layer.γ)

function transfer_sample(layer::Gaussian)
    μ = transfer_mean(layer)
    σ = sqrt.(transfer_var(layer))
    z = randn(promote_type(eltype(μ), eltype(σ)), size(μ))
    return μ .+ σ .* z
end

transfer_mode(layer::Gaussian) = gauss_mode.(layer.θ, layer.γ)
transfer_mean(layer::Gaussian) = transfer_mode(layer)
transfer_var(layer::Gaussian) = inv.(abs.(layer.γ))
transfer_std(layer::Gaussian) = sqrt.(transfer_var(layer))

function transfer_mean_abs(layer::Gaussian)
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    return @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * SpecialFunctions.erf(μ / √(2ν))
end

gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x
gauss_free(θ::Real, γ::Real) = -θ^2 / abs(2γ) + log(abs(γ)/π/2) / 2
gauss_mode(θ::Real, γ::Real) = θ / abs(γ)

function ∂free_energy(layer::Gaussian)
    θ = layer.θ
    γ = abs.(layer.γ)
    return (
        θ = -θ ./ γ,
        γ = sign.(γ) .* (γ .+ θ.^2) ./ γ.^2 / 2
    )
end

struct GaussStats{A<:AbstractArray}
    x1::A
    x2::A
    function GaussStats(layer::AbstractLayer, data::AbstractArray; wts = nothing)
        x1 = batchmean(layer, data; wts)
        x2 = batchmean(layer, data.^2; wts)
        return new{typeof(x1)}(x1, x2)
    end
end

Base.size(stats::GaussStats) = size(stats.x1)
suffstats(layer::Gaussian, x::AbstractArray; wts = nothing) = GaussStats(layer, x; wts)

function ∂energy(layer::Gaussian, stats::GaussStats)
    @assert size(layer) == size(stats)
    return (; θ = -stats.x1, γ = stats.x2 / 2)
end
