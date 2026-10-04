#= Methods shared by every `StandardizedRBM`, including a `CenteredRBM` (the special case
with lazy unit `Trues` scales). The `_maybe_div` / `_maybe_mul` helpers skip the scaling
no-op for lazy unit scales, keeping the centered hot paths free of divisions by one. =#

"""
    RBM(rbm::StandardizedRBM)

Plain `RBM` sharing the layers and weights of `rbm` but ignoring its offsets and scales, so
it is *not* equivalent to `rbm`. For an equivalent model use [`unstandardize`](@ref) (or
[`uncenter`](@ref) for a [`CenteredRBM`](@ref)).
"""
RBM(rbm::StandardizedRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

# same offsets and scales as `rbm`, with `plain` as the underlying RBM
_with_offsets(rbm::StandardizedRBM, plain::RBM) = StandardizedRBM(plain, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)

# scales of the weights of the equivalent plain RBM, shaped like `rbm.w` (lazy for a CenteredRBM)
_scale_w(rbm::StandardizedRBM) = _along_visible(rbm, rbm.scale_v) .* _along_hidden(rbm, rbm.scale_h)

"""
    delta_energy(rbm)

The constant energy shift of `rbm` with respect to its equivalent plain `RBM`.
"""
delta_energy(rbm::RBM) = 0
delta_energy(rbm::StandardizedRBM) = interaction_energy(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h))

potts_to_gumbel(rbm::StandardizedRBM) = _with_offsets(rbm, potts_to_gumbel(RBM(rbm)))
gumbel_to_potts(rbm::StandardizedRBM) = _with_offsets(rbm, gumbel_to_potts(RBM(rbm)))

"""
    zerosum(rbm::StandardizedRBM)

Returns an equivalent model, with the same offsets and scales, whose equivalent plain
`RBM` ([`unstandardize`](@ref)) is in the zerosum gauge. The gauge condition applies to the
plain parameters, since the interaction involves the offset and scaled activations rather
than `v` and `h` themselves. Does nothing without Potts layers.
"""
function zerosum(rbm::StandardizedRBM)
    has_potts_layers(rbm) || return rbm
    return standardize(zerosum(unstandardize(rbm)), rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

# weights of the equivalent plain RBM (`rbm.w` itself for a CenteredRBM)
unstandardized_weights(rbm::StandardizedRBM) = _maybe_div(rbm.w, _scale_w(rbm))

"""
    weight_norms(rbm::StandardizedRBM)

Norms of the unstandardized weights attached to each hidden unit. For the norms of the
standardized weights, use `weight_norms(RBM(rbm))`.
"""
weight_norms(rbm::StandardizedRBM) = weight_norms(RBM(rbm.visible, rbm.hidden, unstandardized_weights(rbm)))

function interaction_energy(rbm::StandardizedRBM, v::AbstractArray, h::AbstractArray)
    return interaction_energy(RBM(rbm), standardize_v(rbm, v), standardize_h(rbm, h))
end

function inputs_h_from_v(rbm::StandardizedRBM, v::AbstractArray)
    inputs = inputs_h_from_v(RBM(rbm), standardize_v(rbm, v))
    return _maybe_div(inputs, rbm.scale_h)
end

function inputs_v_from_h(rbm::StandardizedRBM, h::AbstractArray)
    inputs = inputs_v_from_h(RBM(rbm), standardize_h(rbm, h))
    return _maybe_div(inputs, rbm.scale_v)
end

function free_energy(rbm::StandardizedRBM, v::AbstractArray)
    E = energy(rbm.visible, v)
    inputs = inputs_h_from_v(rbm, v)
    F = -cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_h), inputs)
    return E + F - ΔE
end

function free_energy_h(rbm::StandardizedRBM, h::AbstractArray)
    E = energy(rbm.hidden, h)
    inputs = inputs_v_from_h(rbm, h)
    F = -cgf(rbm.visible, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_v), inputs)
    return E + F - ΔE
end

function ∂interaction_energy(rbm::StandardizedRBM, v::AbstractArray, h::AbstractArray; kwargs...)
    return ∂interaction_energy(RBM(rbm), standardize_v(rbm, v), standardize_h(rbm, h); kwargs...)
end

function log_pseudolikelihood(rbm::StandardizedRBM, v::AbstractArray; kwargs...)
    return log_pseudolikelihood(unstandardize(rbm), v; kwargs...)
end

function ∂regularize!(
        ∂::∂RBM, offset_rbm::StandardizedRBM;
        l2_fields::Real = 0,
        l1_weights::Real = 0,
        l2_weights::Real = 0,
        l2l1_weights::Real = 0,
        regularize_unstandardized::Bool = true,
        zerosum::Bool = false # whether to zerosum gradients
    )
    if !regularize_unstandardized
        # regularization applies directly to the offset model's parameters
        ∂regularize!(∂, RBM(offset_rbm); l2_fields, l1_weights, l2_weights, l2l1_weights)
    elseif !all(iszero, (l2_fields, l1_weights, l2_weights, l2l1_weights))
        # regularization applies to the parameters of the equivalent plain RBM, whose
        # weights are `w / scale_w` and whose visible fields absorb `w * offset_h`
        rbm = unstandardize(offset_rbm)
        scale_w = _scale_w(offset_rbm)
        if !iszero(l2_fields)
            visible_reg = ∂regularize_fields(rbm.visible; l2_fields)
            ∂.visible .+= visible_reg
            # chain rule through the absorbed field shift; only the field rows of
            # `visible_reg` are nonzero, so summing over parameter rows collects them
            field_reg = dropdims(sum(visible_reg; dims = 1); dims = 1)
            ∂.w .-= _maybe_div(field_reg .* _along_hidden(offset_rbm, offset_rbm.offset_h), scale_w)
        end
        _∂regularize_weights!(∂.w, rbm; l1_weights, l2_weights, l2l1_weights, scale = scale_w)
    end
    zerosum && zerosum!(∂, offset_rbm)
    return ∂
end

function regularization_penalty(rbm::StandardizedRBM; regularize_unstandardized::Bool = true, kwargs...)
    return regularization_penalty(regularize_unstandardized ? unstandardize(rbm) : RBM(rbm); kwargs...)
end

"""
    zerosum!(rbm::StandardizedRBM)

In-place version of `zerosum(rbm)`. Offsets and scales are not modified.
"""
function zerosum!(rbm::StandardizedRBM)
    if rbm.visible isa _PottsLayers
        # Gauge move on the weights of the equivalent plain RBM, w̃ = w / (scale_v ⊗ scale_h):
        # subtract their mean over visible colors (scale_h cancels out of the w update).
        ξ = mean(_maybe_div(rbm.w, rbm.scale_v); dims = 1)
        rbm.w .-= _maybe_mul(ξ, rbm.scale_v)
        zerosum!(rbm.visible.θ; dims = 1)
        # Compensate hidden fields. Unlike a plain RBM, the interaction involves
        # v - offset_v, so the color-sum of the visible offsets enters the shift.
        vdims = ntuple(identity, ndims(rbm.visible))
        Ov = sum(rbm.offset_v; dims = 1)
        Δθh = _maybe_div(reshape(sum(ξ .* (1 .- Ov); dims = vdims), size(rbm.hidden)), rbm.scale_h)
        shift_fields!(rbm.hidden, Δθh)
    end
    if rbm.hidden isa _PottsLayers
        scale_h = _along_hidden(rbm, rbm.scale_h)
        ζ = mean(_maybe_div(rbm.w, scale_h); dims = ndims(rbm.visible) + 1)
        rbm.w .-= _maybe_mul(ζ, scale_h)
        zerosum!(rbm.hidden.θ; dims = 1)
        hdims = ntuple(d -> d + ndims(rbm.visible), ndims(rbm.hidden))
        Oh = reshape(sum(rbm.offset_h; dims = 1), map(one, size(rbm.visible))..., 1, size(rbm.hidden)[2:end]...)
        Δθv = _maybe_div(reshape(sum(ζ .* (1 .- Oh); dims = hdims), size(rbm.visible)), rbm.scale_v)
        shift_fields!(rbm.visible, Δθv)
    end
    return rbm
end

"""
    zerosum!(∂, rbm::StandardizedRBM)

Projects the gradient so that it doesn't modify the zerosum gauge of the equivalent
plain `RBM` (see [`unstandardize`](@ref)), with offsets and scales
held fixed. The gauge condition on the weights reads `sum(w ./ scale_v; dims = 1) == 0`
over Potts colors (similarly for hidden Potts with `scale_h`), so the component removed
is the gauge direction `ξ .* scale_v`.
"""
function zerosum!(∂::∂RBM, rbm::StandardizedRBM)
    if rbm.visible isa _PottsLayers
        zerosum!(∂.visible; dims = 2) # dim 1 of `par` is the (singleton) parameter type
        ξ = mean(_maybe_div(∂.w, rbm.scale_v); dims = 1)
        ∂.w .-= _maybe_mul(ξ, rbm.scale_v)
    end
    if rbm.hidden isa _PottsLayers
        zerosum!(∂.hidden; dims = 2)
        scale_h = _along_hidden(rbm, rbm.scale_h)
        ζ = mean(_maybe_div(∂.w, scale_h); dims = ndims(rbm.visible) + 1)
        ∂.w .-= _maybe_mul(ζ, scale_h)
    end
    return ∂
end
