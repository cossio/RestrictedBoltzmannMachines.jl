"""
    energy(layer, x)

Layer energy, reduced over layer dimensions.
"""
function energy(layer, x::AbstractArray)
    E = energies(layer, x)
    return maybe_scalar(sum_(E; dims = layerdims(layer)))
end

function energy(layer::Union{Binary, Spin, Potts}, x::AbstractArray)
    @assert size(x)[1:ndims(layer)] == size(layer)
    if ndims(x) > ndims(layer)
        return -reshape(x, length(layer), :)' * vec(layer.θ)
    else
        return -vec(x)' * vec(layer.θ)
    end
end

"""
    cgf(layer, inputs = 0, β = 1)

Cumulant generating function of layer, reduced over layer dimensions.
"""
function cgf(layer, inputs = 0, β::Real = 1)
    Γ = cgfs(layer, inputs, β)
    return maybe_scalar(sum_(Γ; dims = layerdims(layer)))
end

"""
    sample(layer, inputs = 0, β = 1)

Samples layer configurations conditioned on inputs.
"""
function sample(layer, inputs , β::Real = 1)
    layer_ = effective(layer, inputs, β)
    return sample(layer_)
end

"""
    cgfs(layer, inputs = 0, β = 1)

Cumulant generating function of units in layer (not reduced over layer dimensions).
"""
function cgfs(layer, inputs, β::Real = 1)
    layer_ = effective(layer, inputs, β)
    return cgfs(layer_) / β
end

function mode(layer, inputs, β::Real = 1)
    layer_ = effective(layer, inputs, β)
    return mode(layer_)
end

"""
    energies(layer, x)

Energies of units in layer (not reduced over layer dimensions).
"""
function energies end

energies(layer::Union{Binary, Spin, Potts}, x) = -layer.θ .* x

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
    γ = @. 2l.γp * l.γn / (l.γp + l.γn)
    ξ = @. (l.γn - l.γp) / (l.γp + l.γn - abs(l.γn - l.γp))
    θ = @. (l.θp * l.γn + l.θn * l.γp) / (l.γp + l.γn)
    Δ = @. γ * (l.θp - l.θn) / (l.γp + l.γn)
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
