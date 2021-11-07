"""
    batchmul(fields, x)

Forms the product fields * x, summing over the fields dimension while
leaving the batch dimensions.
"""
function batchmul(fields::AbstractArray, x::AbstractArray)
    @assert size(x) == (size(fields)..., size(x)[end])
    return reshape(x, length(fields), size(x)[end])' * vec(fields)
end

function energy(layer::Union{Binary,Spin,Potts}, x::AbstractArray)
    @assert size(x) == (size(layer)..., size(x)[end])
    return -batchmul(layer.θ, x)
end

const _LayersWithTheta = Union{Binary,Spin,Potts,Gaussian,StdGaussian,ReLU,pReLU}
Base.ndims(layer::_LayersWithTheta) = ndims(layer.θ)
Base.size(layer::_LayersWithTheta) = size(layer.θ)
Base.length(layer::_LayersWithTheta) = length(layer.θ)

layerdims(layer) = ntuple(identity, ndims(layer))

Binary(layer::Union{Spin,Potts}) = Binary(layer.θ)
Spin(layer::Union{Binary,Potts}) = Spin(layer.θ)
Potts(layer::Union{Binary,Spin}) = Potts(layer.θ)

ReLU(layer::Gaussian) = ReLU(layer.θ, layer.γ)
Gaussian(layer::ReLU) = Gaussian(layer.θ, layer.γ)

function dReLU(p::Union{Gaussian, ReLU}, n::Union{Gaussian, ReLU})
    @assert size(p) == size(n)
    return dReLU(p.θ, -n.θ, p.γ, n.γ)
end

function pReLU(l::dReLU)
    θ = @. (l.θp * l.γn + l.θn * l.γp) / (l.γp + l.γn)
    γ = @. 2l.γp * l.γn / (l.γp + l.γn)
    Δ = @. γ * (l.θp - l.θn) / (l.γp + l.γn)
    η = @. (l.γn - l.γp) / (l.γp + l.γn)
    return pReLU(θ, γ, Δ, η)
end

function dReLU(l::pReLU)
    γp = @. l.γ / (1 + l.η)
    γn = @. l.γ / (1 - l.η)
    θp = @. l.θ + l.Δ / (1 + l.η)
    θn = @. l.θ - l.Δ / (1 - l.η)
    return dReLU(θp, θn, γp, γn)
end
