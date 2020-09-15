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
__energy(layer::Gaussian, x::AbstractArray) = @. (abs(layer.γ) * x/2 - layer.θ) * x
__cgf(layer::Gaussian) = @. layer.θ^2 / abs(layer.γ) / 2 - log(abs(layer.γ)/π/2)/2
_random(layer::Gaussian) = randn_like(layer.θ) ./ sqrt.(abs.(layer.γ)) .+ layer.θ ./ abs.(layer.γ)
_transfer_mean(layer::Gaussian) = @. layer.θ / abs(layer.γ)
_transfer_var(layer::Gaussian) = @. inv(abs(layer.γ))
_transfer_std(layer::Gaussian) = sqrt.(_transfer_var(layer))
_transfer_mode(layer::Gaussian) = _transfer_mean(layer)
effective_β(layer::Gaussian, β) = Gaussian(β .* layer.θ, β .* layer.γ)
effective_I(layer::Gaussian, I) = Gaussian(layer.θ .+ I, broadlike(layer.γ, I))
_transfer_pdf(layer::Gaussian, x) = gauss_pdf.(layer.θ, layer.γ, x)
_transfer_cdf(layer::Gaussian, x) = gauss_cdf.(layer.θ, layer.γ, x)
_transfer_logpdf(layer::Gaussian, x) = gauss_logpdf.(layer.θ, layer.γ, x)
_transfer_logcdf(layer::Gaussian, x) = gauss_logcdf.(layer.θ, layer.γ, x)

function gauss_logpdf(θ::Real, γ::Real, x::Real)
    ξ = (x - θ / abs(γ)) * √abs(γ)
    return -ξ^2 / 2 + log(abs(γ) / 2π) / 2
end

function gauss_logcdf(θ::Real, γ::Real, x::Real)
    ξ = (x - θ / abs(γ)) * √abs(γ)
    return logerfc(-ξ / √2) - log(2)
end

gauss_pdf(θ::Real, γ::Real, x::Real) = exp(gauss_logpdf(θ, γ, x))
gauss_cdf(θ::Real, γ::Real, x::Real) = exp(gauss_logcdf(θ, γ, x))

function _transfer_mean_abs(layer::Gaussian)
    μ = _transfer_mean(layer)
    ν = _transfer_var(layer)
    @. √(2ν/π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

#= gradients =#

@adjoint function __energy(layer::Gaussian, x::AbstractArray)
    ∂θ = -x
    ∂γ = @. sign(layer.γ) * x^2/2
    back(Δ) = ((θ = ∂θ .* Δ, γ = ∂γ .* Δ), nothing)
    return __energy(layer, x), back
end

@adjoint function __cgf(layer::Gaussian)
    Γ = __cgf(layer)
    θ, γ = layer.θ, layer.γ
    ∂θ = @. θ / abs(γ)
    ∂γ = @. -(θ^2 + abs(γ)) * sign(γ) / (2abs(γ)^2)
    back(Δ) = ((θ = ∂θ .* Δ, γ = ∂γ .* Δ),)
    return Γ, back
end
