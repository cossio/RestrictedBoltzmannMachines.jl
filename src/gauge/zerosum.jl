function zerosum!(A::AbstractArray; dims=1)
    A .-= mean(A; dims)
    return A
end

"""
    zerosum!(rbm)

Fixes zerosum gauge for `Potts` layers (visible or hidden). If the `rbm` doesn't have
`Potts` layers, does nothing.
"""
function zerosum!(rbm::RBM)
    if rbm.visible isa Potts
        zerosum!(rbm.visible.θ; dims = 1)
        zerosum!(rbm.w; dims = 1)
    end
    if rbm.hidden isa Potts
        zerosum!(rbm.hidden.θ; dims = 1)
        zerosum!(rbm.w; dims = ndims(rbm.visible) + 1)
    end
    return rbm
end
