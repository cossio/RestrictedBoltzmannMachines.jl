export Gaussian

struct Gaussian{T,N} <: AbstractLayer{T,N}
    θ::Array{T,N}
    γ::Array{T,N}
    function Gaussian{T,N}(θ::Array{T,N}, γ::Array{T,N}) where {T,N}
        size(θ) == size(γ) || pardimserror()
        new{T,N}(θ, γ)
    end
end
Gaussian(θ::Array{T,N}, γ::Array{T,N}) where {T,N} = Gaussian{T,N}(θ, γ)
Gaussian{T}(n::Int...) where {T} = Gaussian(zeros(T, n...), ones(T, n...))
Gaussian(n::Int...) = Gaussian{Float64}(n...)
fields(layer::Gaussian) = (layer.θ, layer.γ)
Flux.@functor Gaussian
_random(layer::Gaussian) = randn_like(layer.θ) ./ sqrt.(abs.(layer.γ)) .+ layer.θ ./ abs.(layer.γ)
_transfer_mean(layer::Gaussian) = @. layer.θ / abs(layer.γ)
_transfer_var(layer::Gaussian) = @. inv(abs(layer.γ))
_transfer_std(layer::Gaussian) = sqrt.(_transfer_var(layer))
_transfer_mode(layer::Gaussian) = _transfer_mean(layer)
effective_β(layer::Gaussian, β) = Gaussian(β .* layer.θ, β .* layer.γ)
effective_I(layer::Gaussian, I) = Gaussian(layer.θ .+ I, broadlike(layer.γ, I))

function __transfer_logpdf(layer::Gaussian, x)
    θ, γ = layer.θ, abs.(layer.γ)
    ξ = @. (x - θ / γ) * √γ
    return @. -ξ^2 / 2 + log(γ/2/π) / 2
end

function __transfer_logcdf(layer::Gaussian, x)
    θ, γ = layer.θ, abs.(layer.γ)
    ξ = @. (x - θ / γ) * √γ
    two = convert(eltype(ξ), 2)
    @. logerfc(-ξ / √two) - log(two)
end

function _transfer_mean_abs(layer::Gaussian)
    μ = _transfer_mean(layer)
    ν = _transfer_var(layer)
    @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

function __transfer_entropy(layer::Gaussian)
    ν = _transfer_var(layer)
    return @. log(2ν * π * ℯ)
end

#= gradients =#

__energy(layer::Gaussian, x::AbstractArray) = @. (abs(layer.γ) * x/2 - layer.θ) * x
@adjoint function __energy(layer::Gaussian, x::AbstractArray)
    ∂θ = -x
    ∂γ = @. sign(layer.γ) * x^2/2
    back(Δ) = ((θ = ∂θ .* Δ, γ = ∂γ .* Δ), nothing)
    return __energy(layer, x), back
end

__cgf(layer::Gaussian) = @. layer.θ^2 / abs(layer.γ) / 2 - log(abs(layer.γ)/π/2)/2
@adjoint function __cgf(layer::Gaussian)
    Γ = __cgf(layer)
    θ, γ = layer.θ, layer.γ
    ∂θ = @. θ / abs(γ)
    ∂γ = @. -(θ^2 + abs(γ)) * sign(γ) / (2abs(γ)^2)
    back(Δ) = ((θ = ∂θ .* Δ, γ = ∂γ .* Δ),)
    return Γ, back
end
