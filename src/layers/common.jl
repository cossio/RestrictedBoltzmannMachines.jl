const _ThetaLayers = Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}
Base.size(layer::_ThetaLayers) = size(layer.θ)
Base.size(layer::_ThetaLayers, d::Int) = size(layer.θ, d)
Base.length(layer::_ThetaLayers) = length(layer.θ)

"""
    energies(layer, x)

Energies of units in layer (not reduced over layer dimensions).
"""
function energies(layer::Union{Binary, Spin, Potts}, x::AbstractTensor)
    check_size(layer, x)
    return -layer.θ .* x
end

function energy(layer::Union{Binary, Spin, Potts}, x::AbstractTensor)
    check_size(layer, x)
    xconv = activations_convert_maybe(layer.θ, x)
    E = -flatten(layer, xconv)' * vec(layer.θ)
    return E::Union{Number, AbstractVector}
end

number_of_colors(layer::Potts) = layer.q
number_of_colors(layer) = 1

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
    ξp = @. (1 + abs(layer.ξ)) / (1 + max(2layer.ξ, 0))
    ξn = @. (1 + abs(layer.ξ)) / (1 - min(2layer.ξ, 0))
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

dReLU(layer::Gaussian) = dReLU(layer.θ, layer.θ, layer.γ, layer.γ)
pReLU(layer::Gaussian) = pReLU(dReLU(layer))
xReLU(layer::Gaussian) = xReLU(dReLU(layer))

#dReLU(layer::ReLU) = dReLU(layer.θ, zero(layer.θ), layer.γ, inf.(layer.γ))

# function pReLU(layer::ReLU)
#     θ = layer.θ
#     γ = 2layer.γ
#     η = one.(layer.γ)
#     Δ = zero.(layer.θ)
#     return pReLU(θ, γ, Δ, η)
# end

# function xReLU(layer::ReLU)
#     θ = layer.θ
#     γ = 2layer.γ
#     ξ = inf.(layer.γ)
#     Δ = zero.(layer.θ)
#     return xReLU(θ, γ, Δ, ξ)
# end
