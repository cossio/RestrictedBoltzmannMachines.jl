export dReLU2

struct dReLU2{T,N} <: AbstractLayer{T,N}
    θ::Array{T,N}
    Δ::Array{T,N}
    γ::Array{T,N}
    η::Array{T,N}
    function dReLU2{T,N}(ps::Vararg{Array{T,N}, 4}) where {T,N}
        allequal(size.(ps)...) || pardimserror()
        return new{T,N}(ps...)
    end
end
dReLU2(ps::Vararg{Array{T,N}, 4}) where {T,N} = dReLU2{T,N}(ps...)
function dReLU2(l::dReLU)
    θ = @. (l.γn * l.θp + l.γp * l.θn) / (l.γp + l.γn)
    Δ = @. 2l.γp * l.γn * (l.θp - l.θn) / (l.γp + l.γn)^2
    γ = @. 2l.γp * l.γn / (l.γp + l.γn)
    η = @. (l.γn - l.γp) / (l.γp + l.γn)
    return dReLU2(θ, Δ, γ, η)
end
function dReLU(l::dReLU2)
    γp = @. l.γ / (1 + l.η)
    γn = @. l.γ / (1 - l.η)
    θp = @. l.θ + l.Δ / (1 + l.η)
    θn = @. l.θ - l.Δ / (1 - l.η)
    return dReLU(θp, θn, γp, γn)
end
dReLU2(p::Union{ReLU,Gaussian}, n::Union{ReLU,Gaussian}) = dReLU2(dReLU(p, n))
dReLU2{T}(n::Int...) where {T} = dReLU2(dReLU{T}(n...))
dReLU2(n::Int...) = dReLU2{Float64}(n...)
Flux.@functor dReLU2
fields(l::dReLU2) = (l.θ, l.Δ, l.γ, l.η)
relus_pair(layer::dReLU2) = relus_pair(dReLU(layer))
gauss_pair(layer::dReLU2) = gauss_pair(dReLU(layer))
probs_pair(layer::dReLU2) = probs_pair(dReLU(layer))
__energy(layer::dReLU2, x::AbstractArray) = __energy(dReLU(layer), x)
__cgf(layer::dReLU2) = __cgf(dReLU(layer))
_random(layer::dReLU2) = _random(dReLU(layer))
_transfer_mode(layer::dReLU2) = _transfer_mode(dReLU(layer))
_transfer_mean(layer::dReLU2) = _transfer_mean(dReLU(layer))
_transfer_std(layer::dReLU2) = _transfer_std(dReLU(layer))
_transfer_var(layer::dReLU2) = _transfer_var(dReLU(layer))
_transfer_mean_abs(layer::dReLU2) = _transfer_mean_abs(dReLU(layer))
effective_β(layer::dReLU2, β) = dReLU2(effective_β(dReLU(layer), β))
effective_I(layer::dReLU2, β) = dReLU2(effective_I(dReLU(layer), β))
