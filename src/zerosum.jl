"""
    zerosum!(rbm)

If the `rbm` has `Potts` layers (visible or hidden), fixes zerosum gauge on the parameters.
Otherwise, does nothing.
"""
function zerosum!(rbm::AbstractRBM)
    zerosum_visible!(rbm)
    zerosum_hidden!(rbm)
    return rbm
end

zerosum_visible!(rbm::RBM{<:AbstractLayer, <:AbstractLayer}) = rbm
zerosum_hidden!(rbm::RBM{<:AbstractLayer, <:AbstractLayer})  = rbm

function zerosum_visible!(rbm::RBM{<:Potts, <:AbstractLayer})
    zerosum!(visible(rbm))
    zerosum!(weights(rbm); dims = 1)
    return nothing
end

function zerosum_hidden!(rbm::RBM{<:AbstractLayer, <:Potts})
    zerosum!(hidden(rbm))
    zerosum!(weights(rbm); dims = 1 + ndims(visible(rbm)))
    return nothing
end

function zerosum!(rbm::RBM{<:Potts, <:Potts})
    zerosum!(visible(rbm))
    zerosum!(hidden(rbm))
    zerosum!(weights(rbm); dims = 1)
    zerosum!(weights(rbm); dims = 1 + ndims(visible(rbm)))
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
