export ReLU

struct ReLU{T,N} <: AbstractLayer{T,N}
    θ::Array{T,N}
    γ::Array{T,N}
    function ReLU{T,N}(θ::Array{T,N}, γ::Array{T,N}) where {T,N}
        size(θ) == size(γ) || pardimserror()
        new{T,N}(θ, γ)
    end
end
ReLU(θ::Array{T,N}, γ::Array{T,N}) where {T,N} = ReLU{T,N}(θ, γ)
ReLU{T}(n::Int...) where {T} = ReLU(zeros(T, n...), ones(T, n...))
ReLU(n::Int...) = ReLU{Float64}(n...)
ReLU(layer::Gaussian) = ReLU(layer.θ, layer.γ)
ReLU{T,N}(layer::Gaussian{T,N}) where {T,N} = ReLU(layer)
Gaussian(layer::ReLU) = Gaussian(layer.θ, layer.γ)
Gaussian{T,N}(layer::ReLU{T,N}) where {T,N} = Gaussian(layer.θ, layer.γ)
fields(layer::ReLU) = (layer.θ, layer.γ)
Flux.@functor ReLU

function __energy(layer::ReLU, x::AbstractArray)
    E = __energy(Gaussian(layer), x)
    return @. ifelse(x < 0, inf(E), E)
end

__cgf(layer::ReLU) = relu_cgf.(layer.θ, layer.γ)
_random(layer::ReLU) = relu_rand.(layer.θ, layer.γ)

function _transfer_mode(layer::ReLU)
    x = _transfer_mode(Gaussian(layer))
    return @. max(x, zero(x))
end

effective_β(layer::ReLU, β) = ReLU(effective_β(Gaussian(layer), β))
effective_I(layer::ReLU, I) = ReLU(effective_I(Gaussian(layer), I))
relu_cgf(θ::Real, γ::Real) = logerfcx(-θ/√(2abs(γ))) - log(2abs(γ)/π)/2
exp_relu_cgf(θ::Real, γ::Real) = erfcx(-θ/√(2abs(γ))) / sqrt(2abs(γ)/π)

function ∇relu_cgf(θ::Numeric, γ::Numeric)
    #all(γ .> 0) || throw(ArgumentError("All γ must be positive"))
    Γ = @. relu_cgf(θ, γ)
    dθ = @. (θ + inv(exp(Γ))) / abs(γ)
    dγ = @. -(one(θ) + θ * dθ) / (2γ)
    return Γ, dθ, dγ
end
@adjoint function relu_cgf(θ::Real, γ::Real)
    z, dθ, dγ = ∇relu_cgf(θ, γ)
    return z, Δ -> (Δ * dθ, Δ * dγ)
end
@adjoint function broadcasted(::typeof(relu_cgf), θ::Numeric, γ::Numeric)
    z, dθ, dγ = ∇relu_cgf(θ, γ)
    return z, Δ -> (nothing, Δ .* dθ, Δ .* dγ)
end

function relu_rand(θ::Real, γ::Real)
    μ = θ / abs(γ)
    σ = √inv(abs(γ))
    return randnt_half(μ, σ)
end
function ∇relu_rand(θ::Numeric, γ::Numeric) # ∇(survival) / pdf (implicit grads for ReLU samples)
    μ = @. θ / abs(γ)
    σ = @. inv(√abs(γ))
    z, dμ, dσ = ∇randnt_half(μ, σ)
    dθ = @. dμ / abs(γ)
    dγ = @. (-μ * dμ - σ * dσ / 2) / γ
    return z, dθ, dγ
end
@adjoint function relu_rand(θ::Real, γ::Real)
    z, dθ, dγ = ∇relu_rand(θ, γ)
    return z, Δ -> (Δ * dθ, Δ * dγ)
end
@adjoint function broadcasted(::typeof(relu_rand), θ::Numeric, γ::Numeric)
    z, dθ, dγ = ∇relu_rand(θ, γ)
    return z, Δ -> (nothing, Δ .* dθ, Δ .* dγ)
end

"""
    relu_mills(θ, γ, x)

Returns the Mills ratio for the unit.
The Mills ratio is defined as (1 - CDF(x)) / PDF(x).
"""
function relu_mills(θ::Real, γ::Real, x::Real)
    γ = abs(γ)
    return √(π/(2γ)) * erfcx(-(θ - x * γ) / √(2γ))
end

"""
    relu_survival(θ, γ, x)

The survival function is defined as 1 - CDF(x).
"""
function relu_survival(θ::Real, γ::Real, x::Real)
    γ = abs(γ)
    result = erfc(-(θ - x * γ) / √(2γ)) / erfc(-θ / √(2γ))
    return ifelse(x < 0, one(result), result)
end

function relu_logsurvival(θ::Real, γ::Real, x::Real)
    γ = abs(γ)
    result = logerfc(-(θ - x * γ) / √(2γ)) - logerfc(-θ / √(2γ))
    return ifelse(x < 0, zero(result), result)
end

relu_cdf(θ::Real, γ::Real, x::Real) = 1 - relu_survival(θ, γ, x)
relu_pdf(θ::Real, γ::Real, x::Real) = exp(relu_logpdf(θ, γ, x))
relu_logcdf(θ::Real, γ::Real, x::Real) = log1p(-relu_survival(θ, γ, x))

function relu_logpdf(θ::Real, γ::Real, x::Real)
    result = -(abs(γ) * x/2 - θ) * x - relu_cgf(θ, γ)
    return ifelse(x < 0, -inf(result), result)
end

function _transfer_mean(layer::ReLU)
    g = Gaussian(layer)
    μ = _transfer_mean(g)
    σ = _transfer_std(g)
    @. μ + σ * tnmean(-μ/σ)
end

function _transfer_std(layer::ReLU)
    g = Gaussian(layer)
    μ = _transfer_mean(g)
    σ = _transfer_std(g)
    return @. σ * tnstd(-μ/σ)
end

function _transfer_var(layer::ReLU)
    g = Gaussian(layer)
    μ = _transfer_mean(g)
    σ = _transfer_std(g)
    return @. σ^2 * tnvar(-μ/σ)
end

_transfer_mean_abs(layer::ReLU) = _transfer_mean(layer)
__transfer_logpdf(layer::ReLU, x) = relu_logpdf.(layer.θ, layer.γ, x)
__transfer_logcdf(layer::ReLU, x) = relu_logcdf.(layer.θ, layer.γ, x)