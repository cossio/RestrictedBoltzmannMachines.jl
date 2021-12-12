"""
    zerosum!(rbm)

If the `rbm` has `Potts` layers (visible or hidden), fixes zerosum gauge on the weights
and on the layer parameters.
Otherwise, does nothing.
"""
zerosum!(rbm::RBM) = rbm

function zerosum!(rbm::RBM{<:Potts, <:Any})
    zerosum!(rbm.visible)
    zerosum!(rbm.weights; dims = 1)
    return rbm
end

function zerosum!(rbm::RBM{<:Any, <:Potts})
    zerosum!(rbm.hidden)
    zerosum!(rbm.weights; dims = 1 + ndims(rbm.visible))
    return RBM
end

function zerosum!(rbm::RBM{<:Potts, <:Potts})
    zerosum!(rbm.visible)
    zerosum!(rbm.hidden)
    zerosum!(rbm.weights; dims = 1)
    zerosum!(rbm.weights; dims = 1 + ndims(rbm.visible))
    return rbm
end

function zerosum!(layer::Potts)
    zerosum!(layer.θ; dims=1)
    return layer
end

function zerosum!(A::AbstractArray; dims=1)
    A .-= mean(A; dims=dims)
    return A
end
