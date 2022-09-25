"""
    ReLU(θ, γ)

Layer with ReLU units, with location parameters `θ` and scale parameters `γ`.
"""
struct ReLU{N,A} <: AbstractLayer{N}
    par::A
    function ReLU{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 2 # θ, γ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

ReLU(par::AbstractArray) = ReLU{ndims(par) - 1, typeof(par)}(par)

function ReLU(; θ, γ)
    par = vstack((θ, γ))
    return ReLU(par)
end

function ReLU(::Type{T}, sz::Dims) where {T}
    θ = zeros(T, sz)
    γ = ones(T, sz)
    return ReLU(; θ, γ)
end

ReLU(sz::Dims) = ReLU(Float64, sz)
Base.propertynames(::ReLU) = (:θ, :γ)

function Base.getproperty(layer::ReLU, name::Symbol)
    if name === :θ
        return @view getfield(layer, :par)[1, ..]
    elseif name === :γ
        return @view getfield(layer, :par)[2, ..]
    else
        return getfield(layer, name)
    end
end

function energies(layer::ReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return relu_energy.(layer.θ, layer.γ, x)
end

cfgs(layer::ReLU, inputs = 0) = relu_cfg.(layer.θ .+ inputs, layer.γ)
sample_from_inputs(layer::ReLU, inputs = 0) = relu_rand.(layer.θ .+ inputs, layer.γ)
mode_from_inputs(layer::ReLU, inputs = 0) = max.((layer.θ .+ inputs) ./ abs.(layer.γ), 0)
mean_abs_from_inputs(layer::ReLU, inputs = 0) = mean_from_inputs(layer, inputs)
std_from_inputs(layer::ReLU, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))

function mean_from_inputs(layer::ReLU, inputs = 0)
    g = Gaussian(layer.par)
    μ = mean_from_inputs(g, inputs)
    σ = sqrt.(var_from_inputs(g, inputs))
    return @. μ + σ * tnmean(-μ / σ)
end

function var_from_inputs(layer::ReLU, inputs = 0)
    g = Gaussian(layer.par)
    μ = mean_from_inputs(g, inputs)
    ν = var_from_inputs(g, inputs)
    return @. ν * tnvar(-μ / √ν)
end

function meanvar_from_inputs(layer::ReLU, inputs = 0)
    g = Gaussian(layer.par)
    μ = mean_from_inputs(g, inputs)
    ν = var_from_inputs(g, inputs)
    σ = sqrt.(ν)
    tμ, tν = tnmeanvar(-μ ./ σ)
    return μ + σ .* tμ, ν .* tν
end

function ∂cfgs(layer::ReLU, inputs = 0)
    μ, ν = meanvar_from_inputs(layer, inputs)
    ∂θ = -μ
    ∂γ = sign.(layer.γ) .* (ν .+ μ.^2) / 2
    return vstack((∂θ, ∂γ))
end

function ∂energy_from_moments(layer::ReLU, moments::AbstractArray)
    return ∂energy_from_moments(Gaussian(layer.par), moments)
end

function moments_from_samples(layer::ReLU, data::AbstractArray; wts = nothing)
    return moments_from_samples(Gaussian(layer.par), data; wts)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    return x < 0 ? inf(E) : E
end

function relu_cfg(θ::Real, γ::Real)
    abs_γ = abs(γ)
    return -logerfcx(-θ / √(2abs_γ)) + log(2abs_γ/π) / 2
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end
