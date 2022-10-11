"""
    Gaussian(θ, γ)

Gaussian layer, with location parameters `θ` and scale parameters `γ`.
"""
struct Gaussian{N,A} <: AbstractLayer{N}
    par::A
    function Gaussian{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 2 # θ, γ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

Gaussian(par::AbstractArray) = Gaussian{ndims(par) - 1, typeof(par)}(par)

function Gaussian(; θ, γ)
    par = vstack((θ, γ))
    return Gaussian(par)
end

function Gaussian(::Type{T}, sz::Dims) where {T}
    θ = zeros(T, sz)
    γ = ones(T, sz)
    return Gaussian(; θ, γ)
end

Gaussian(sz::Dims) = Gaussian(Float64, sz)
Base.propertynames(::Gaussian) = (:θ, :γ)

function Base.getproperty(layer::Gaussian, name::Symbol)
    if name === :θ
        return @view getfield(layer, :par)[1, ..]
    elseif name === :γ
        return @view getfield(layer, :par)[2, ..]
    else
        return getfield(layer, name)
    end
end

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)
gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x

cgfs(layer::Gaussian, inputs = 0) = gauss_cfg.(layer.θ .+ inputs, layer.γ)
gauss_cfg(θ::Real, γ::Real) = -θ^2 / abs(2γ) + log(abs(γ)/π/2) / 2

function sample_from_inputs(layer::Gaussian, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    σ = std_from_inputs(layer, inputs)
    z = randn!(similar(μ))
    return μ .+ σ .* z
end

mean_from_inputs(l::Gaussian, inputs = 0) = (l.θ .+ inputs) ./ abs.(l.γ)
var_from_inputs(l::Gaussian, inputs = 0) = inv.(abs.(l.γ .+ zero(inputs)))
std_from_inputs(l::Gaussian, inputs = 0) = sqrt.(var_from_inputs(l, inputs))
mode_from_inputs(l::Gaussian, inputs = 0) = mean_from_inputs(l, inputs)

function meanvar_from_inputs(l::Gaussian, inputs = 0)
    return (mean_from_inputs(l, inputs), var_from_inputs(l, inputs))
end

function mean_abs_from_inputs(layer::Gaussian, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    ν = var_from_inputs(layer, inputs)
    return @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

function ∂cgfs(layer::Gaussian, inputs = 0)
    ∂θ = @. -(layer.θ .+ inputs) / abs(layer.γ)
    ∂γ = @. inv(layer.γ) / 2 + sign(layer.γ) * ∂θ^2 / 2
    return vstack((∂θ, ∂γ))
end

function ∂energy_from_moments(layer::Gaussian, moments::AbstractArray)
    @assert size(layer.par) == size(moments)
    x1 = @view moments[1, ..]
    x2 = @view moments[2, ..]
    ∂θ = -x1
    ∂γ = @. sign(layer.γ) * x2 / 2
    return vstack((∂θ, ∂γ))
end

function moments_from_samples(layer::Gaussian, data::AbstractArray; wts = nothing)
    x1 = batchmean(layer, data; wts)
    x2 = batchmean(layer, data.^2; wts)
    return vstack((x1, x2))
end
