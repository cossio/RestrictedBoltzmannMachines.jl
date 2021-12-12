"""
    sample_from_inputs(layer, inputs, β = 1)

Samples layer configurations conditioned on inputs.
"""
function sample_from_inputs end

function energy(layer::Union{Binary, Spin, Potts}, x::AbstractArray)
    @assert size(x) == (size(layer)..., size(x)[end])
    return -reshape(x, length(layer), size(x)[end])' * vec(layer.θ)
end

const _ThetaLayers = Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}
Base.ndims(layer::_ThetaLayers) = ndims(layer.θ)
Base.size(layer::_ThetaLayers) = size(layer.θ)
Base.size(layer::_ThetaLayers, d::Int) = size(layer.θ, d)
Base.length(layer::_ThetaLayers) = length(layer.θ)

layerdims(layer) = ntuple(identity, ndims(layer))

function pReLU(l::dReLU)
    γ = @. 2l.γp * l.γn / (l.γp + l.γn)
    η = @. (l.γn - l.γp) / (l.γp + l.γn)
    θ = @. (l.θp * l.γn + l.θn * l.γp) / (l.γp + l.γn)
    Δ = @. γ * (l.θp - l.θn) / (l.γp + l.γn)
    return pReLU(θ, Δ, γ, η)
end

function dReLU(l::pReLU)
    γp = @. l.γ / (1 + l.η)
    γn = @. l.γ / (1 - l.η)
    θp = @. l.θ + l.Δ / (1 + l.η)
    θn = @. l.θ - l.Δ / (1 - l.η)
    return dReLU(θp, θn, γp, γn)
end

function xReLU(l::dReLU)
    #= The xReLU <-> dReLU conversion is bijective only if the γ's are positive =#
    γp = abs.(l.γp)
    γn = abs.(l.γn)
    γ = @. 2γp * γn / (γp + γn)
    ξ = @. (γn - γp) / (γp + γn - abs(γn - γp))
    θ = @. (l.θp * γn + l.θn * γp) / (γp + γn)
    Δ = @. γ * (l.θp - l.θn) / (γp + γn)
    return xReLU(θ, Δ, γ, ξ)
end

function dReLU(l::xReLU)
    ξp = @. (1 + abs(l.ξ)) / (1 + max( 2l.ξ, 0))
    ξn = @. (1 + abs(l.ξ)) / (1 + max(-2l.ξ, 0))
    γp = @. l.γ * ξp
    γn = @. l.γ * ξn
    θp = @. l.θ + l.Δ * ξp
    θn = @. l.θ - l.Δ * ξn
    return dReLU(θp, θn, γp, γn)
end

function xReLU(l::pReLU)
    ξ = @. l.η / (1 - abs(l.η))
    return xReLU(l.θ, l.Δ, l.γ, ξ)
end

function pReLU(l::xReLU)
    η = @. l.ξ / (1 + abs(l.ξ))
    return pReLU(l.θ, l.Δ, l.γ, η)
end
