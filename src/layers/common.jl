Base.size(layer::Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}) = size(layer.θ)
Base.length(layer::Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}) = length(layer.θ)

const _FieldLayers = Union{Binary, Spin, Potts}

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

function ∂cfgs(layer::_FieldLayers, inputs::Union{Real,AbstractArray} = 0)
    return (; θ = -mean_from_inputs(layer, inputs))
end

struct FieldStats{A<:AbstractArray}
    μ::A
    function FieldStats(layer::_FieldLayers, x::AbstractArray; wts=nothing)
        μ = batchmean(layer, x; wts)
        return new{typeof(μ)}(μ)
    end
end
Base.size(stats::FieldStats) = size(stats.μ)
Base.ndims(stats::FieldStats) = size(stats.μ)

suffstats(layer::_FieldLayers, data::AbstractArray; wts = nothing) = FieldStats(layer, data; wts)

function ∂energy(layer::_FieldLayers, stats::FieldStats)
    @assert size(layer) == size(stats.μ)
    return (; θ = -stats.μ)
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
    abs_γp = abs.(layer.γp)
    abs_γn = abs.(layer.γn)
    γ = @. 2abs_γp * abs_γn / (abs_γp + abs_γn)
    η = @. (abs_γn - abs_γp) / (abs_γp + abs_γn)
    θ = @. (layer.θp * abs_γn + layer.θn * abs_γp) / (abs_γp + abs_γn)
    Δ = @. γ * (layer.θp - layer.θn) / (abs_γp + abs_γn)
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
    abs_γp = abs.(layer.γp)
    abs_γn = abs.(layer.γn)
    γ = @. 2abs_γp * abs_γn / (abs_γp + abs_γn)
    ξ = @. (abs_γn - abs_γp) / (abs_γp + abs_γn - abs(abs_γn - abs_γp))
    θ = @. (layer.θp * abs_γn + layer.θn * abs_γp) / (abs_γp + abs_γn)
    Δ = @. γ * (layer.θp - layer.θn) / (abs_γp + abs_γn)
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
