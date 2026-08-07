# Statistics recovered from a moments array (the layout of `moments_from_samples`
# and `moments_from_inputs`: first axis = moment index, trailing batch dimensions
# allowed and preserved).

"""
    mean_from_moments(layer, moments)

Mean unit activations `<x>` from a moments array.
"""
mean_from_moments(::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU}, moments::AbstractArray) = moments[1, ..]
mean_from_moments(::Union{dReLU, pReLU, xReLU, nsReLU}, moments::AbstractArray) = moments[1, ..] + moments[2, ..]

"""
    var_from_moments(layer, moments)

Variance of unit activations from a moments array.
"""
var_from_moments(::Union{Binary, Potts, PottsGumbel}, moments::AbstractArray) = moments[1, ..] .* (1 .- moments[1, ..])
var_from_moments(::Spin, moments::AbstractArray) = (1 .- moments[1, ..]) .* (1 .+ moments[1, ..])
var_from_moments(::Union{Gaussian, ReLU}, moments::AbstractArray) = moments[2, ..] - moments[1, ..] .^ 2

function var_from_moments(::Union{dReLU, pReLU, xReLU, nsReLU}, moments::AbstractArray)
    # xp and xn cannot be nonzero simultaneously, so <x^2> = <xp^2> + <xn^2>
    return moments[3, ..] + moments[4, ..] - (moments[1, ..] + moments[2, ..]) .^ 2
end

"""
    batchmean_moments(layer, moments; wts = nothing)

Average a per-configuration moments array (as returned by `moments_from_inputs`
with batched inputs) over its batch dimensions, weighted by `wts`.
"""
function batchmean_moments(layer::AbstractLayer, moments::AbstractArray; wts = nothing)
    m = wmean(moments; wts, dims = (ndims(layer) + 2):ndims(moments))
    return reshape(m, ntuple(d -> size(moments, d), Val(ndims(layer) + 1)))
end
