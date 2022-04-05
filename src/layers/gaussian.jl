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

Base.repeat(l::Gaussian, n::Int...) = Gaussian(repeat(l.θ, n...), repeat(l.γ, n...))

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

transfer_mean(l::Gaussian) = l.θ ./ abs.(l.γ)
transfer_var(l::Gaussian) = inv.(abs.(l.γ))
transfer_meanvar(l::Gaussian) = transfer_mean(l), transfer_var(l)
transfer_std(l::Gaussian) = sqrt.(transfer_var(l))
transfer_mode(l::Gaussian) = transfer_mean(l)

function transfer_mean_abs(layer::Gaussian)
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    return @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x
gauss_free(θ::Real, γ::Real) = -θ^2 / abs(2γ) + log(abs(γ)/π/2) / 2

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
