Base.size(layer::Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}) = size(layer.θ)
Base.length(layer::Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}) = length(layer.θ)

const _FieldLayers = Union{Binary, Spin, Potts}

Base.propertynames(::_FieldLayers) = (:θ,)

function Base.getproperty(layer::_FieldLayers, name::Symbol)
    if name === :θ
        return @view getfield(layer, :par)[1, ..]
    else
        return getfield(layer, name)
    end
end

"""
    energies(layer, x)

Energies of units in layer (not reduced over layer dimensions).
"""
function energies(layer::_FieldLayers, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return -layer.θ .* x
end

function energy(layer::_FieldLayers, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    xconv = activations_convert_maybe(layer.θ, x)
    if ndims(layer) == ndims(x)
        return -dot(layer.θ, x)
    else
        Eflat = -vec(layer.θ)' * flatten(layer, xconv)
        return reshape(Eflat, batch_size(layer, x))
    end
end

function ∂cfgs(layer::_FieldLayers, inputs = 0)
    ∂θ = -mean_from_inputs(layer, inputs)
    return vstack((∂θ,))
end

function ∂energy_from_moments(layer::_FieldLayers, moments::AbstractArray)
    @assert size(layer.par) == size(moments)
    x1 = moments[1, ..]
    ∂θ = -x1
    return vstack((∂θ,))
end

function moments_from_samples(layer::_FieldLayers, data::AbstractArray; wts = nothing)
    x1 = batchmean(layer, data; wts)
    return vstack((x1,))
end

"""
    colors(layer)

Number of possible states of units in discrete layers.
"""
colors(layer::Union{Spin,Binary}) = 2
colors(layer::Potts) = size(layer, 1)

"""
    sitedims(layer)

Number of dimensions of layer, with special handling of Potts layer,
for which the first dimension doesn't count as a site dimension.
"""
sitedims(layer::AbstractLayer) = ndims(layer)
sitedims(layer::Potts) = ndims(layer) - 1

"""
    sitesize(layer)

Size of layer, with special handling of Potts layer,
for which the first dimension doesn't count as a site dimension.
"""
sitesize(layer::AbstractLayer) = size(layer)
sitesize(layer::Potts) = size(layer)[2:end]

function pReLU(layer::dReLU)
    γ = @. 2abs(layer.γp) * abs(layer.γn) / (abs(layer.γp) + abs(layer.γn))
    η = @. (abs(layer.γn) - abs(layer.γp)) / (abs(layer.γp) + abs(layer.γn))
    θ = @. (layer.θp * abs(layer.γn) + layer.θn * abs(layer.γp)) / (abs(layer.γp) + abs(layer.γn))
    Δ = @. γ * (layer.θp - layer.θn) / (abs(layer.γp) + abs(layer.γn))
    return pReLU(; θ, γ, Δ, η)
end

function dReLU(layer::pReLU)
    γp = @. layer.γ / (1 + layer.η)
    γn = @. layer.γ / (1 - layer.η)
    θp = @. layer.θ + layer.Δ / (1 + layer.η)
    θn = @. layer.θ - layer.Δ / (1 - layer.η)
    return dReLU(; θp, θn, γp, γn)
end

function xReLU(layer::dReLU)
    γ = @. 2abs(layer.γp) * abs(layer.γn) / (abs(layer.γp) + abs(layer.γn))
    ξ = @. (abs(layer.γn) - abs(layer.γp)) / (abs(layer.γp) + abs(layer.γn) - abs(abs(layer.γn) - abs(layer.γp)))
    θ = @. (layer.θp * abs(layer.γn) + layer.θn * abs(layer.γp)) / (abs(layer.γp) + abs(layer.γn))
    Δ = @. γ * (layer.θp - layer.θn) / (abs(layer.γp) + abs(layer.γn))
    return xReLU(; θ, γ, Δ, ξ)
end

function dReLU(layer::xReLU)
    ξp = @. (1 + abs(layer.ξ)) / (1 + max(2layer.ξ, 0))
    ξn = @. (1 + abs(layer.ξ)) / (1 - min(2layer.ξ, 0))
    γp = @. layer.γ * ξp
    γn = @. layer.γ * ξn
    θp = @. layer.θ + layer.Δ * ξp
    θn = @. layer.θ - layer.Δ * ξn
    return dReLU(; θp, θn, γp, γn)
end

function xReLU(layer::pReLU)
    ξ = @. layer.η / (1 - abs(layer.η))
    return xReLU(; layer.θ, layer.γ, layer.Δ, ξ)
end

function pReLU(layer::xReLU)
    η = @. layer.ξ / (1 + abs(layer.ξ))
    return pReLU(; layer.θ, layer.γ, layer.Δ, η)
end

dReLU(layer::Gaussian) = dReLU(; θp = layer.θ, θn = layer.θ, γp = layer.γ, γn = layer.γ)
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

function moments_from_samples(layer::Union{pReLU,xReLU}, data::AbstractArray; wts = nothing)
    xp = max.(data, 0)
    xn = min.(data, 0)
    xp1 = batchmean(layer, xp; wts)
    xn1 = batchmean(layer, xn; wts)
    xp2 = batchmean(layer, xp.^2; wts)
    xn2 = batchmean(layer, xn.^2; wts)
    return vstack((xp1, xn1, xp2, xn2))
end
