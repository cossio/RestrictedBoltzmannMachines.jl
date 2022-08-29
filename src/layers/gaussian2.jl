"""
    Gaussian(θ, γ)

Gaussian layer, with location parameters `θ` and scale parameters `γ`.
"""
struct Gaussian{N,A} <: AbstractLayer{N}
    par::A
    function Gaussian(par::AbstractArray)
        @assert size(par, 1) == 2 # θ, γ
        N = ndims(par) - 1
        return new{N, typeof(par)}(par)
    end
end

function Gaussian(::Type{T}, n::Int...) where {T}
    θ = zeros(T, 1, n...)
    γ = ones(T, 1, n...)
    par = cat(θ, γ; dims=1)
    return Gaussian(par)
end

Gaussian(n::Int...) = Gaussian(Float64, n...)

function Base.getproperty(layer::Gaussian, name::Symbol)
    if name === :θ
        return @view layer.par[1, ..]
    elseif name === :γ
        return @view layer.par[2, ..]
    else
        return getfield(layer, name)
    end
end

Base.propertynames(::Gaussian2) = (:θ, :γ)

Base.repeat(l::Gaussian, n::Int...) = Gaussian(repeat(l.par, 1, n...))

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)

function cfgs(layer::Gaussian, inputs = 0)
    return @. -(layer.θ .+ inputs)^2 / abs(2layer.γ) + log(abs(layer.γ)/π/2) / 2
end

function sample_from_inputs(layer::Gaussian, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    σ = std_from_inputs(layer, inputs)
    z = randn!(similar(μ))
    return μ .+ σ .* z
end

mean_from_inputs(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = (l.θ .+ inputs) ./ abs.(l.γ)
var_from_inputs(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = inv.(abs.(l.γ .+ zero(inputs)))
std_from_inputs(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = sqrt.(var_from_inputs(l, inputs))
mode_from_inputs(l::Gaussian, inputs::Union{Real,AbstractArray} = 0) = mean_from_inputs(l, inputs)

function meanvar_from_inputs(l::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    return mean_from_inputs(l, inputs), var_from_inputs(l, inputs)
end

function mean_abs_from_inputs(layer::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    μ = mean_from_inputs(layer, inputs)
    ν = var_from_inputs(layer, inputs)
    return @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x

function moments_from_inputs(layer::Gaussian, inputs = 0)
    θ = layer.θ .+ inputs
    abs_γ = abs.(layer.γ)
    ∂θ = -θ ./ abs_γ
    ∂γ = @. sign(layer.γ) * (abs_γ + θ^2) / abs_γ^2 / 2
    reshape(∂θ, 1, size(∂θ)...)
end

function stack_left(A::AbstractArray...)
    Y = map(A) do x
        reshape(x, 1, size(x)...)
    end
    return cat(Y..., dims=1)
end

function ∂cfgs(layer::Gaussian, inputs::Union{Real,AbstractArray} = 0)
    θ = layer.θ .+ inputs
    abs_γ = abs.(layer.γ)
    return (
        θ = -θ ./ abs_γ,
        γ = @. sign(layer.γ) * (abs_γ + θ^2) / abs_γ^2 / 2
    )
end

function ∂energy(layer::Gaussian, stats::GaussStats)
    @assert size(layer) == size(stats)
    return (; θ = -stats.x1, γ = sign.(layer.γ) .* stats.x2 / 2)
end
