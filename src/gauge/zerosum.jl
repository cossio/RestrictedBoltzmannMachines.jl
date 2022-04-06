"""
    zerosum!(rbm)

Fixes zerosum gauge for `Potts` layers (visible or hidden). If the `rbm` doesn't have
`Potts` layers, does nothing.
"""
function zerosum!(rbm::RBM)
    zerosum_visible!(rbm)
    zerosum_hidden!(rbm)
    return rbm
end

function zerosum_visible!(rbm::RBM{<:Potts, <:Any})
    zerosum!(visible(rbm))
    zerosum!(weights(rbm); dims = 1)
    return nothing
end

zerosum_visible!(rbm::RBM) = rbm

function zerosum_hidden!(rbm::RBM{<:Any, <:Potts})
    zerosum!(hidden(rbm))
    zerosum!(weights(rbm); dims = 1 + ndims(visible(rbm)))
    return nothing
end

zerosum_hidden!(rbm::RBM) = rbm

function zerosum!(layer::Potts)
    zerosum!(layer.θ; dims=1)
    return nothing
end

function zerosum!(A::AbstractArray; dims=1)
    A .-= mean(A; dims)
    return A
end
