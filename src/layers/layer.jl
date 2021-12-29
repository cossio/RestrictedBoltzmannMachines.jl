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
function cgf(layer, inputs = 0, β::Real = true)
    Γ = cgfs(layer, inputs, β)
    return maybe_scalar(sum_(Γ; dims = layerdims(layer)))
end

"""
    transfer_sample(layer, inputs = 0, β = 1)

Samples layer configurations conditioned on inputs.
"""
function transfer_sample(layer, inputs , β::Real = true)
    layer_ = effective(layer, inputs, β)
    return transfer_sample(layer_)
end

"""
    transfer_mode(layer, inputs = 0)

Mode of unit activations.
"""
function transfer_mode(layer, inputs)
    layer_ = effective(layer, inputs)
    return transfer_mode(layer_)
end

"""
    transfer_mean(layer, inputs = 0, β = 1)

Mean of unit activations.
"""
function transfer_mean(layer, inputs, β::Real = true)
    layer_ = effective(layer, inputs, β)
    return transfer_mean(layer_)
end

"""
    transfer_var(layer, inputs = 0, β = 1)

Variance of unit activations.
"""
function transfer_var(layer, inputs, β::Real = true)
    layer_ = effective(layer, inputs, β)
    return transfer_var(layer_)
end

"""
    transfer_mean_abs(layer, inputs = 0, β = 1)

Mean of absolute value of unit activations.
"""
function transfer_mean_abs(layer, inputs, β::Real = true)
    layer_ = effective(layer, inputs, β)
    return transfer_mean_abs(layer_)
end
"""
    conjugates(layer, inputs = 0, β = 1)

Unit activation moments, conjugate to layer parameters.
These are obtained by differentiating the `cgf` with respect to the layer parameters.
"""
function conjugates(layer, inputs, β::Real = true)
    layer_ = effective(layer, inputs, β)
    return conjugates(layer_)
end

"""
    cgfs(layer, inputs = 0, β = 1)

Cumulant generating function of units in layer (not reduced over layer dimensions).
"""
function cgfs(layer, inputs, β::Real = true)
    layer_ = effective(layer, inputs, β)
    return cgfs(layer_) / β
end

"""
    energies(layer, x)

Energies of units in layer (not reduced over layer dimensions).
"""
energies(layer::Union{Binary, Spin, Potts}, x) = -layer.θ .* x

const _ThetaLayers = Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}
Base.ndims(layer::_ThetaLayers) = ndims(layer.θ)
Base.size(layer::_ThetaLayers) = size(layer.θ)
Base.size(layer::_ThetaLayers, d::Int) = size(layer.θ, d)
Base.length(layer::_ThetaLayers) = length(layer.θ)

layerdims(layer) = ntuple(identity, ndims(layer))

function pReLU(layer::dReLU)
    γ = @. 2layer.γp * layer.γn / (layer.γp + layer.γn)
    η = @. (layer.γn - layer.γp) / (layer.γp + layer.γn)
    θ = @. (layer.θp * layer.γn + layer.θn * layer.γp) / (layer.γp + layer.γn)
    Δ = @. γ * (layer.θp - layer.θn) / (layer.γp + layer.γn)
    return pReLU(θ, γ, Δ, η)
end

function dReLU(layer::pReLU)
    γp = @. layer.γ / (1 + layer.η)
    γn = @. layer.γ / (1 - layer.η)
    θp = @. layer.θ + layer.Δ / (1 + layer.η)
    θn = @. layer.θ - layer.Δ / (1 - layer.η)
    return dReLU(θp, θn, γp, γn)
end

function xReLU(layer::dReLU)
    γ = @. 2layer.γp * layer.γn / (layer.γp + layer.γn)
    ξ = @. (layer.γn - layer.γp) / (layer.γp + layer.γn - abs(layer.γn - layer.γp))
    θ = @. (layer.θp * layer.γn + layer.θn * layer.γp) / (layer.γp + layer.γn)
    Δ = @. γ * (layer.θp - layer.θn) / (layer.γp + layer.γn)
    return xReLU(θ, γ, Δ, ξ)
end

function dReLU(layer::xReLU)
    ξp = @. (1 + abs(layer.ξ)) / (1 + max( 2layer.ξ, 0))
    ξn = @. (1 + abs(layer.ξ)) / (1 + max(-2layer.ξ, 0))
    γp = @. layer.γ * ξp
    γn = @. layer.γ * ξn
    θp = @. layer.θ + layer.Δ * ξp
    θn = @. layer.θ - layer.Δ * ξn
    return dReLU(θp, θn, γp, γn)
end

function xReLU(layer::pReLU)
    ξ = @. layer.η / (1 - abs(layer.η))
    return xReLU(layer.θ, layer.γ, layer.Δ, ξ)
end

function pReLU(layer::xReLU)
    η = @. layer.ξ / (1 + abs(layer.ξ))
    return pReLU(layer.θ, layer.γ, layer.Δ, η)
end
