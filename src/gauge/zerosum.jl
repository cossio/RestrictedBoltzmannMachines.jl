zerosum(A::AbstractArray; dims = 1) = A .- mean(A; dims)
zerosum!(A::AbstractArray; dims = 1) = A .= zerosum(A; dims)

"""
    zerosum!(rbm)

Fixes zerosum gauge for `Potts` layers (visible or hidden). If the `rbm` doesn't have
`Potts` layers, does nothing.
"""
function zerosum!(rbm::RBM)
    if rbm.visible isa Potts
        zerosum!(rbm.visible.θ; dims = 1)
    end
    if rbm.hidden isa Potts
        zerosum!(rbm.hidden.θ; dims = 1)
    end
    zerosum_weights!(rbm.w, rbm)
    return rbm
end

"""
    zerosum(rbm)

Fixes zerosum gauge for `Potts` layers (visible or hidden). If the `rbm` doesn't have
`Potts` layers, does nothing.
"""
function zerosum(rbm::RBM)
    if rbm.visible isa Potts
        visible = Potts(zerosum(rbm.visible.θ; dims = 1))
    else
        visible = rbm.visible
    end
    if rbm.hidden isa Potts
        hidden = Potts(zerosum(rbm.hidden.θ; dims = 1))
    else
        hidden = rbm.hidden
    end
    w = zerosum_weights(rbm.w, rbm)
    return RBM(visible, hidden, w)
end

function zerosum_weights(weights::AbstractArray, rbm::RBM)
    @assert size(weights) == size(rbm.w)
    if rbm.visible isa Potts
        w = zerosum(weights; dims = 1)
    else
        w = weights
    end
    if rbm.hidden isa Potts
        return zerosum(w; dims = ndims(rbm.visible) + 1)
    else
        return w
    end
end

zerosum_weights!(w::AbstractArray, rbm::RBM) = w .= zerosum_weights(w, rbm)

"""
    zerosum!(∂, rbm)

Enforces zerosum gauge on the RBM gradient.
"""
function zerosum!(∂::NamedTuple, rbm::RBM)
    if rbm.visible isa Potts
        zerosum!(∂.visible.θ; dims = 1)
    end
    if rbm.hidden isa Potts
        zerosum!(∂.hidden.θ; dims = 1)
    end
    zerosum_weights!(∂.w, rbm)
    return ∂
end
