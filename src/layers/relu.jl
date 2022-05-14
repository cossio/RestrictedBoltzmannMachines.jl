"""
    ReLU(θ, γ)

ReLU layer, with location parameters `θ` and scale parameters `γ`.
"""
struct ReLU{N,A1,A2} <: AbstractLayer{N}
    θ::A1
    γ::A2
    function ReLU(θ::AbstractArray, γ::AbstractArray)
        @assert size(θ) == size(γ)
        return new{ndims(θ), typeof(θ), typeof(γ)}(θ, γ)
    end
end

ReLU(::Type{T}, sz::Int...) where {T} = ReLU(zeros(T, sz...), ones(T, sz...))
ReLU(sz::Int...) = ReLU(Float64, sz...)

Base.repeat(l::ReLU, n::Int...) = ReLU(repeat(l.θ, n...), repeat(l.γ, n...))

function energies(layer::ReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return relu_energy.(layer.θ, layer.γ, x)
end

free_energies(layer::ReLU, inputs::Union{Real,AbstractArray} = 0) = relu_free.(layer.θ .+ inputs, layer.γ)
transfer_sample(layer::ReLU, inputs::Union{Real,AbstractArray} = 0) = relu_rand.(layer.θ .+ inputs, layer.γ)
transfer_mode(layer::ReLU, inputs::Union{Real,AbstractArray} = 0) = max.((layer.θ .+ inputs) ./ abs.(layer.γ), 0)

function transfer_mean(layer::ReLU, inputs::Union{Real,AbstractArray} = 0)
    g = Gaussian(layer.θ, layer.γ)
    μ = transfer_mean(g, inputs)
    σ = sqrt.(var_from_inputs(g, inputs))
    return @. μ + σ * tnmean(-μ / σ)
end

transfer_mean_abs(layer::ReLU, inputs::Union{Real,AbstractArray} = 0) = transfer_mean(layer, inputs)

function var_from_inputs(layer::ReLU, inputs::Union{Real,AbstractArray} = 0)
    g = Gaussian(layer.θ, layer.γ)
    μ = transfer_mean(g, inputs)
    ν = var_from_inputs(g, inputs)
    return @. ν * tnvar(-μ / √ν)
end

function meanvar_from_inputs(layer::ReLU, inputs::Union{Real,AbstractArray} = 0)
    g = Gaussian(layer.θ, layer.γ)
    μ = transfer_mean(g, inputs)
    ν = var_from_inputs(g, inputs)
    σ = sqrt.(ν)
    tμ, tν = tnmeanvar(-μ ./ σ)
    return μ + σ .* tμ, ν .* tν
end

std_from_inputs(layer::ReLU, inputs::Union{Real,AbstractArray} = 0) = sqrt.(var_from_inputs(layer, inputs))

function ∂free_energies(layer::ReLU, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    ν = var_from_inputs(layer, inputs)
    return (θ = -μ, γ = sign.(layer.γ) .* (ν .+ μ.^2) / 2)
end

const ReLUStats = GaussStats
suffstats(layer::ReLU, data::AbstractArray; wts = nothing) = ReLUStats(layer, data; wts)

function ∂energy(layer::ReLU, stats::ReLUStats)
    @assert size(layer) == size(stats)
    return (; θ = -stats.x1, γ = sign.(layer.γ) .* stats.x2 / 2)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    return x < 0 ? inf(E) : E
end

function relu_free(θ::Real, γ::Real)
    abs_γ = abs(γ)
    return -logerfcx(-θ / √(2abs_γ)) + log(2abs_γ/π) / 2
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end
