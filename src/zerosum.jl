"""
    zerosum!(rbm)

If the `rbm` has `Potts` layers (visible or hidden), fixes zerosum gauge on the weights
and on the layer fields.
Otherwise, does nothing.
"""
function zerosum!(rbm::RBM)
    _zerosum_visible!(rbm)
    _zerosum_hidden!(rbm)
    return rbm
end

_zerosum_visible!(rbm::RBM{<:AbstractLayer, <:AbstractLayer}) = rbm
_zerosum_hidden!(rbm::RBM{<:AbstractLayer, <:AbstractLayer})  = rbm

function _zerosum_visible!(rbm::RBM{<:Potts, <:AbstractLayer})
    zerosum!(rbm.visible)
    zerosum!(rbm.w; dims = 1)
    return nothing
end

function _zerosum_hidden!(rbm::RBM{<:AbstractLayer, <:Potts})
    zerosum!(rbm.hidden)
    zerosum!(rbm.w; dims = 1 + ndims(rbm.visible))
    return nothing
end

function zerosum!(rbm::RBM{<:Potts, <:Potts})
    zerosum!(rbm.visible)
    zerosum!(rbm.hidden)
    zerosum!(rbm.w; dims = 1)
    zerosum!(rbm.w; dims = 1 + ndims(rbm.visible))
    return nothing
end

function zerosum!(layer::Potts)
    zerosum!(layer.θ; dims=1)
    return nothing
end

function zerosum!(A::AbstractArray; dims=1)
    A .-= Statistics.mean(A; dims=dims)
    return A
end
