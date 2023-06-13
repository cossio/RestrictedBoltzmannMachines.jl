zerosum(A::AbstractArray; dims = 1) = A .- mean(A; dims)
zerosum!(A::AbstractArray; dims = 1) = A .= zerosum(A; dims)

"""
    zerosum(rbm)

Returns an equivalent `rbm` in zerosum gauge. Only affects Potts layers. If the `rbm`
doesn't have `Potts` layers, does nothing.
"""
function zerosum(rbm::RBM)
    vdims = ntuple(identity, ndims(rbm.visible))
    hdims = ntuple(d -> d + ndims(rbm.visible), ndims(rbm.hidden))
    if rbm.visible isa Potts && rbm.hidden isa Potts
        ωv = mean(rbm.w; dims = 1)
        ωh = mean(rbm.w; dims = 1 + ndims(rbm.visible))
        ω = mean(rbm.w; dims = (1, 1 + ndims(rbm.visible)))
        visible = Potts(; θ = rbm.visible.θ .- mean(rbm.visible.θ; dims = 1) .+ reshape(sum(ωh .- ω; dims = hdims), size(rbm.visible)))
        hidden = Potts(; θ = rbm.hidden.θ .- mean(rbm.hidden.θ; dims = 1) .+ reshape(sum(ωv .- ω; dims = vdims), size(rbm.hidden)))
        return oftype(rbm, RBM(visible, hidden, rbm.w .- ωv .- ωh .+ ω))
    elseif rbm.visible isa Potts
        ωv = mean(rbm.w; dims = 1)
        visible = Potts(; θ = rbm.visible.θ .- mean(rbm.visible.θ; dims = 1))
        hidden = shift_fields(rbm.hidden, reshape(sum(ωv; dims = vdims), size(rbm.hidden)))
        return oftype(rbm, RBM(visible, hidden, rbm.w .- ωv))
    elseif rbm.hidden isa Potts
        ωh = mean(rbm.w, dims = ndims(rbm.visible) + 1)
        visible = shift_fields(rbm.visible, reshape(sum(ωh; dims = hdims), size(rbm.visible)))
        hidden = Potts(; θ = rbm.hidden.θ .- mean(rbm.hidden.θ; dims = 1))
        return oftype(rbm, RBM(visible, hidden, rbm.w .- ωh))
    else
        # if the RBM doesn't have Potts layers, do nothing
        return rbm
    end
end

"""
    zerosum!(rbm)

In-place zero-sum gauge on `rbm`.
"""
function zerosum!(rbm::RBM)
    if rbm.visible isa Potts
        zerosum!(rbm.visible.θ; dims = 1)
        ωv = mean(rbm.w; dims = 1)
        rbm.w .-= ωv
        dims = ntuple(identity, ndims(rbm.visible))
        shift_fields!(rbm.hidden, reshape(sum(ωv; dims), size(rbm.hidden)))
    end
    if rbm.hidden isa Potts
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
    if rbm.visible isa Potts
        zerosum!(∂.visible; dims = 1)
        zerosum!(∂.w; dims = 1)
    end
    if rbm.hidden isa Potts
        zerosum!(∂.hidden; dims = 1)
        zerosum!(∂.w; dims = ndims(rbm.visible) + 1)
    end
    return ∂
end

function zerosum_weights(weights::AbstractArray, rbm::RBM)
    @assert size(weights) == size(rbm.w)
    if rbm.visible isa Potts && rbm.hidden isa Potts
        ωv = mean(weights; dims = 1)
        ωh = mean(weights; dims = 1 + ndims(rbm.visible))
        ω = mean(weights; dims = (1, 1 + ndims(rbm.visible)))
        return oftype(weights, weights .- ωv .- ωh .+ ω)
    elseif rbm.visible isa Potts
        ωv = mean(weights; dims = 1)
        return oftype(weights, weights .- ωv)
    elseif rbm.hidden isa Potts
        ωh = mean(weights, dims = ndims(rbm.visible) + 1)
        return oftype(weights, rbm.w .- ωh)
    else
        # if the RBM doesn't have Potts layers, do nothing
        return weights
    end
end
