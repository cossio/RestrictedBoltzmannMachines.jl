const _FieldLayers = Union{Binary, Spin, Potts, PottsGumbel}

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
    if ndims(layer) == ndims(x)
        return -dot(layer.θ, x)
    else
        xconv = with_eltype_of(layer.θ, x)
        Eflat = -vec(layer.θ)' * flatten(layer, xconv)
        return reshape(Eflat, batch_size(layer, x))
    end
end

function moments_from_inputs(layer::_FieldLayers, inputs::AbstractArray = Falses(size(layer)))
    x1 = mean_from_inputs(layer, inputs)
    return stack([x1]; dims = 1)
end

function ∂energy_from_moments(layer::_FieldLayers, moments::AbstractArray)
    @assert ntuple(d -> size(moments, d), ndims(layer.par)) == size(layer.par)
    x1 = moments[1, ..]
    ∂θ = -x1
    return stack([∂θ]; dims = 1)
end

"""
    moments_from_samples(layer::Union{Binary, Spin, Potts, PottsGumbel}, data; wts = nothing)

One moment slot: `<x>`.
"""
function moments_from_samples(layer::_FieldLayers, data::AbstractArray; wts = nothing)
    x1 = batchmean(layer, data; wts)
    return stack([x1]; dims = 1)
end

"""
    colors(layer)

Number of possible states of units in discrete layers.
"""
colors(layer::Union{Spin, Binary}) = 2
colors(layer::Union{Potts, PottsGumbel}) = size(layer, 1)

"""
    sitedims(layer)

Number of dimensions of layer, with special handling of Potts layer,
for which the first dimension doesn't count as a site dimension.
"""
sitedims(layer::AbstractLayer) = ndims(layer)
sitedims(layer::Union{Potts, PottsGumbel}) = ndims(layer) - 1

"""
    sitesize(layer)

Size of layer, with special handling of Potts layer,
for which the first dimension doesn't count as a site dimension.
"""
sitesize(layer::AbstractLayer) = size(layer)
sitesize(layer::Union{Potts, PottsGumbel}) = size(layer)[2:end]

PottsGumbel(layer::Potts) = PottsGumbel(layer.par)
Potts(layer::PottsGumbel) = Potts(layer.par)

function pReLU(layer::dReLU)
    γ = @. 2abs(layer.γp) * abs(layer.γn) / (abs(layer.γp) + abs(layer.γn))
    η = @. (abs(layer.γn) - abs(layer.γp)) / (abs(layer.γp) + abs(layer.γn))
    θ = @. (layer.θp * abs(layer.γn) + layer.θn * abs(layer.γp)) / (abs(layer.γp) + abs(layer.γn))
    Δ = @. γ * (layer.θp - layer.θn) / (abs(layer.γp) + abs(layer.γn))
    return pReLU(; θ, γ, Δ, η)
end

function dReLU(layer::pReLU)
    _validate_layer_parameters(layer)
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
    _validate_layer_parameters(layer)
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

"""
    moments_from_samples(layer::Union{dReLU, pReLU, xReLU, nsReLU}, data; wts = nothing)

Four moment slots: `<xp>`, `<xn>`, `<xp^2>`, `<xn^2>`, where `xp = max(x, 0)`
and `xn = min(x, 0)`.
"""
function moments_from_samples(layer::Union{dReLU, pReLU, xReLU, nsReLU}, data::AbstractArray; wts = nothing)
    xp = max.(data, false)
    xn = min.(data, false)
    xp1 = batchmean(layer, xp; wts)
    xn1 = batchmean(layer, xn; wts)
    # square through float: narrow-integer samples would overflow in their own type
    xp2 = batchmean(layer, abs2.(float.(xp)); wts)
    xn2 = batchmean(layer, abs2.(float.(xn)); wts)
    return stack([xp1, xn1, xp2, xn2]; dims = 1)
end

function moments_from_inputs(layer::Union{pReLU, xReLU, nsReLU}, inputs::AbstractArray = Falses(size(layer)))
    return moments_from_inputs(dReLU(layer), inputs)
end

mean_from_moments(::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU}, moments::AbstractArray) = moments[1, ..]
mean_from_moments(::Union{dReLU, pReLU, xReLU, nsReLU}, moments::AbstractArray) = moments[1, ..] + moments[2, ..]

var_from_moments(::Union{Binary, Potts, PottsGumbel}, moments::AbstractArray) = moments[1, ..] .* (1 .- moments[1, ..])
var_from_moments(::Union{Gaussian, ReLU}, moments::AbstractArray) = moments[2, ..] - moments[1, ..] .^ 2

function var_from_moments(::Union{dReLU, pReLU, xReLU, nsReLU}, moments::AbstractArray)
    # xp and xn cannot be nonzero simultaneously, so <x^2> = <xp^2> + <xn^2>
    return moments[3, ..] + moments[4, ..] - (moments[1, ..] + moments[2, ..]) .^ 2
end
