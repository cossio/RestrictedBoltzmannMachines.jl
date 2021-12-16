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

_zerosum_visible!(rbm::RBM) = rbm
_zerosum_hidden!(rbm::RBM)  = rbm

function _zerosum_visible!(rbm::RBM{<:Potts, <:Any})
    zerosum!(rbm.visible)
    zerosum!(rbm.weights; dims = 1)
    return nothing
end

function _zerosum_hidden!(rbm::RBM{<:Any, <:Potts})
    zerosum!(rbm.hidden)
    zerosum!(rbm.weights; dims = 1 + ndims(rbm.visible))
    return nothing
end

function zerosum!(layer::Potts)
    zerosum!(layer.θ; dims=1)
    return nothing
end

function zerosum!(A::AbstractArray; dims=1)
    A .-= mean(A; dims=dims)
    return A
end
