struct RBM{V,H,W} <: AbstractRBM{V,H,W}
    visible::V
    hidden::H
    w::W
    """
        RBM(visible, hidden, w)

    Creates a Restricted Boltzmann machine with `visible` and `hidden` layers and weights `w`.
    """
    function RBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
        @assert size(w) == (size(visible)..., size(hidden)...)
        return new{typeof(visible), typeof(hidden), typeof(w)}(visible, hidden, w)
    end
end

visible(rbm::RBM) = rbm.visible
hidden(rbm::RBM) = rbm.hidden
weights(rbm::RBM) = rbm.w

"""
    interaction_energy(rbm, v, h)

Weight mediated interaction energy.
"""
function interaction_energy(rbm::AbstractRBM, v::AbstractArray, h::AbstractArray)
    wflat = flat_w(rbm)
    vflat = flat_v(rbm, v)
    hflat = flat_h(rbm, h)
    bsz = batch_size(rbm, v, h)
    if ndims(visible(rbm)) == ndims(v) || ndims(hidden(rbm)) == ndims(h)
        return reshape_maybe(-vflat' * wflat * hflat, bsz)
    elseif size(vflat, 1) ≥ size(hflat, 1)
        return reshape(-sum((wflat' * vflat) .* hflat; dims=1), bsz)
    else
        return reshape(-sum(vflat .* (wflat * hflat); dims=1), bsz)
    end
end

"""
    inputs_v_to_h(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_v_to_h(rbm::AbstractRBM, v::AbstractArray)
    wflat = flat_w(rbm)
    vflat = activations_convert_maybe(wflat, flat_v(rbm, v))
    iflat = wflat' * vflat
    return reshape(iflat, size(hidden(rbm))..., batch_size(visible(rbm), v)...)
end

"""
    inputs_h_to_v(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_h_to_v(rbm::AbstractRBM, h::AbstractArray)
    wflat = flat_w(rbm)
    hflat = activations_convert_maybe(wflat, flat_h(rbm, h))
    iflat = wflat * hflat
    return reshape(iflat, size(visible(rbm))..., batch_size(hidden(rbm), h)...)
end

"""
    mirror(rbm)

Returns a new RBM with viible and hidden layers flipped.
"""
function mirror(rbm::RBM)
    p(i::Int) = i ≤ ndims(visible(rbm)) ? i + ndims(hidden(rbm)) : i - ndims(visible(rbm))
    perm = ntuple(p, ndims(weights(rbm)))
    w = permutedims(weights(rbm), perm)
    return RBM(hidden(rbm), visible(rbm), w)
end

function ∂interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray; wts = nothing)
    bsz = batch_size(rbm, v, h)
    if ndims(visible(rbm)) == ndims(v) && ndims(hidden(rbm)) == ndims(h)
        wts::Nothing
        ∂wflat = -vec(v) * vec(h)'
    elseif ndims(visible(rbm)) == ndims(v)
        ∂wflat = -vec(v) * vec(batchmean(hidden(rbm), h; wts))'
    elseif ndims(hidden(rbm)) == ndims(h)
        ∂wflat = -vec(batchmean(visible(rbm), v; wts)) * vec(h)'
    else
        hflat = flatten(hidden(rbm), h)
        vflat = activations_convert_maybe(hflat, flatten(visible(rbm), v))
        @assert isnothing(wts) || size(wts) == batch_size(visible(rbm), v)
        if isnothing(wts)
            ∂wflat = -vflat * hflat' / size(vflat, 2)
        else
            @assert size(wts) == bsz
            @assert batch_size(visible(rbm), v) == batch_size(hidden(rbm), h) == size(wts)
            ∂wflat = -vflat * Diagonal(vec(wts)) * hflat' / length(wts)
        end
    end
    ∂w = reshape(∂wflat, size(weights(rbm)))
    return ∂w
end
