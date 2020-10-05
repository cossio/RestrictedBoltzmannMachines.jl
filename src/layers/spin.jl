export Spin

struct Spin{T,N} <: AbstractLayer{T,N}
    θ::Array{T,N}
end
Spin{T}(n::Int...) where {T} = Spin(zeros(T, n...))
Spin(n::Int...) = Spin{Float64}(n...)
fields(layer::Spin) = (layer.θ,)
Flux.@functor Spin
effective_β(layer::Spin, β) = Spin(β .* layer.θ)
effective_I(layer::Spin, I) = Spin(layer.θ .+ I)
_transfer_mode(layer::Spin) = @. ifelse(layer.θ > 0, one(layer.θ), -one(layer.θ))
__cgf(layer::Spin) = @. logaddexp(layer.θ, -layer.θ)
_transfer_mean(layer::Spin) = tanh.(layer.θ)
_transfer_std(layer::Spin) = sech.(layer.θ)
_transfer_var(layer::Spin) = _transfer_std(layer).^2
_transfer_mean_abs(layer::Spin) = ones(eltype(layer.θ), size(layer.θ))
__transfer_logpdf(layer::Spin, x) = spin_logpdf.(layer.θ, x)
spin_logpdf(θ::Real, x::Real) = logsigmoid(2θ * x)

function _random(layer::Spin)
    pinv = @. one(layer.θ) + exp(-2layer.θ)
    u = rand_like(pinv)
    @. ifelse(u * pinv ≤ 1, one(layer.θ), -one(layer.θ))
end
