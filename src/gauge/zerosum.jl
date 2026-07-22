zerosum(A::AbstractArray; dims = 1) = A .- mean(A; dims)
zerosum!(A::AbstractArray; dims = 1) = A .= zerosum(A; dims)

# zerosum only affects Potts layers
has_potts_layers(rbm) =
    rbm.visible isa Union{Potts, PottsGumbel} || rbm.hidden isa Union{Potts, PottsGumbel}

"""
    zerosum(rbm)

Returns an equivalent `rbm` in zerosum gauge. Only affects Potts layers. If the `rbm`
doesn't have `Potts` layers, does nothing.
"""
function zerosum(rbm::RBM)
    has_potts_layers(rbm) || return rbm
    return zerosum!(deepcopy(rbm))
end

"""
    zerosum!(rbm)

In-place zero-sum gauge on `rbm`.
"""
function zerosum!(rbm::RBM)
    if rbm.visible isa Union{Potts, PottsGumbel}
        zerosum!(rbm.visible.θ; dims = 1)
        ωv = mean(rbm.w; dims = 1)
        rbm.w .-= ωv
        dims = ntuple(identity, ndims(rbm.visible))
        shift_fields!(rbm.hidden, reshape(sum(ωv; dims), size(rbm.hidden)))
    end
    if rbm.hidden isa Union{Potts, PottsGumbel}
        zerosum!(rbm.hidden.θ; dims = 1)
        ωh = mean(rbm.w; dims = 1 + ndims(rbm.visible))
        rbm.w .-= ωh
        dims = ntuple(d -> d + ndims(rbm.visible), ndims(rbm.hidden))
        shift_fields!(rbm.visible, reshape(sum(ωh; dims), size(rbm.visible)))
    end
    return rbm
end

"""
    zerosum!(∂, rbm)

Projects the gradient so that it doesn't modify the zerosum gauge.
"""
function zerosum!(∂::∂RBM, rbm::RBM)
    # ∂.visible and ∂.hidden have the layer `par` layout, where dim 1 indexes the
    # parameter type (θ, singleton for Potts) and dim 2 the Potts colors.
    if rbm.visible isa Union{Potts, PottsGumbel}
        zerosum!(∂.visible; dims = 2)
        zerosum!(∂.w; dims = 1)
    end
    if rbm.hidden isa Union{Potts, PottsGumbel}
        zerosum!(∂.hidden; dims = 2)
        zerosum!(∂.w; dims = ndims(rbm.visible) + 1)
    end
    return ∂
end

function zerosum_weights(weights::AbstractArray, rbm::RBM)
    @assert size(weights) == size(rbm.w)
    w = weights
    if rbm.visible isa Union{Potts, PottsGumbel}
        w = w .- mean(w; dims = 1)
    end
    if rbm.hidden isa Union{Potts, PottsGumbel}
        w = w .- mean(w; dims = ndims(rbm.visible) + 1)
    end
    return oftype(weights, w)
end
