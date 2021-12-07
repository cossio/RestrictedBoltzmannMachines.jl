function energy(layer::Union{Binary, Spin, Potts}, x::AbstractArray)
    @assert size(x) == (size(layer)..., size(x)[end])
    return -reshape(x, length(layer), size(x)[end])' * vec(layer.θ)
end

const _ThetaLayers = Union{Binary, Spin, Potts, Gaussian, StdGaussian, ReLU, pReLU}
Base.ndims(layer::_ThetaLayers) = ndims(layer.θ)
Base.size(layer::_ThetaLayers) = size(layer.θ)
Base.size(layer::_ThetaLayers, d::Int) = size(layer.θ, d)
Base.length(layer::_ThetaLayers) = length(layer.θ)

layerdims(layer) = ntuple(identity, ndims(layer))

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
