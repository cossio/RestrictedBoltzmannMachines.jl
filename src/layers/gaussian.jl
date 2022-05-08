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

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)

function free_energies(l::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    return @. -(l.θ .+ inputs)^2 / abs(2l.γ) + log(abs(l.γ)/π/2) / 2
end

function transfer_sample(layer::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    σ = sqrt.(transfer_var(layer, inputs))
    z = randn!(similar(μ))
    return μ .+ σ .* z
end

transfer_mean(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = (l.θ .+ inputs) ./ abs.(l.γ)
transfer_var(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = inv.(abs.(l.γ .+ zero(inputs)))
transfer_std(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = sqrt.(transfer_var(l, inputs))
transfer_mode(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = transfer_mean(l, inputs)

function transfer_meanvar(l::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    return transfer_mean(l, inputs), transfer_var(l, inputs)
end

function transfer_mean_abs(layer::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    ν = transfer_var(layer, inputs)
    return @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x

function ∂free_energies(layer::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    θ = layer.θ .+ inputs
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
