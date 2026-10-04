#= Shared implementations for `CenteredRBM` and `StandardizedRBM`.

Both models subtract offsets from the layer activations entering the interaction energy;
`StandardizedRBM` additionally divides by scales. A `CenteredRBM` therefore behaves like a
`StandardizedRBM` with unit scales, which the `_scale_v` / `_scale_h` accessors expose as
`Ones` so that each method can be written once. The `_maybe_div` / `_maybe_mul` helpers
skip the scaling no-op in the `CenteredRBM` case, keeping its hot paths free of spurious
divisions by one. =#

const OffsetRBM = Union{CenteredRBM, StandardizedRBM}

"""
    RBM(rbm::Union{CenteredRBM, StandardizedRBM})

Plain `RBM` sharing the layers and weights of `rbm` but ignoring its offsets and scales, so
it is *not* equivalent to `rbm`. For an equivalent model use [`uncenter`](@ref) or
[`unstandardize`](@ref).
"""
RBM(rbm::OffsetRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

# same type as `rbm`, with `plain` as the underlying RBM and the offsets (and scales) of `rbm`
_with_offsets(rbm::CenteredRBM, plain::RBM) = CenteredRBM(plain, rbm.offset_v, rbm.offset_h)
_with_offsets(rbm::StandardizedRBM, plain::RBM) = StandardizedRBM(plain, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
# the model equivalent to `plain`, with the offsets (and scales) of `rbm`
_reoffset(rbm::CenteredRBM, plain::RBM) = center(plain, rbm.offset_v, rbm.offset_h)
_reoffset(rbm::StandardizedRBM, plain::RBM) = standardize(plain, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)

standardize_v(rbm::CenteredRBM, v::AbstractArray) = v .- rbm.offset_v
standardize_h(rbm::CenteredRBM, h::AbstractArray) = h .- rbm.offset_h

_scale_v(rbm::StandardizedRBM) = rbm.scale_v
_scale_h(rbm::StandardizedRBM) = rbm.scale_h
_scale_v(rbm::CenteredRBM) = Ones{eltype(rbm.w)}(size(rbm.visible))
_scale_h(rbm::CenteredRBM) = Ones{eltype(rbm.w)}(size(rbm.hidden))

# scales of the weights of the equivalent plain RBM, shaped like `rbm.w`
_scale_w(rbm::StandardizedRBM) = _along_visible(rbm, rbm.scale_v) .* _along_hidden(rbm, rbm.scale_h)
_scale_w(rbm::CenteredRBM) = Ones{eltype(rbm.w)}(size(rbm.w))

# equivalent plain `RBM` modeling the same distribution
_equivalent_rbm(rbm::CenteredRBM) = uncenter(rbm)
_equivalent_rbm(rbm::StandardizedRBM) = unstandardize(rbm)

"""
    delta_energy(rbm)

The constant energy shift of `rbm` with respect to its equivalent plain `RBM`.
"""
delta_energy(rbm::RBM) = 0
delta_energy(rbm::OffsetRBM) = interaction_energy(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h))

potts_to_gumbel(rbm::OffsetRBM) = _with_offsets(rbm, potts_to_gumbel(RBM(rbm)))
gumbel_to_potts(rbm::OffsetRBM) = _with_offsets(rbm, gumbel_to_potts(RBM(rbm)))

"""
    zerosum(rbm::Union{CenteredRBM, StandardizedRBM})

Returns an equivalent model, with the same offsets (and scales), whose equivalent plain
`RBM` ([`uncenter`](@ref) / [`unstandardize`](@ref)) is in the zerosum gauge. The gauge
condition applies to the plain parameters, since the interaction involves the offset (and
scaled) activations rather than `v` and `h` themselves. Does nothing without Potts layers.
"""
function zerosum(rbm::OffsetRBM)
    has_potts_layers(rbm) || return rbm
    return _reoffset(rbm, zerosum(_equivalent_rbm(rbm)))
end

# weights of the equivalent plain RBM (`rbm.w` itself for a CenteredRBM)
unstandardized_weights(rbm::OffsetRBM) = _maybe_div(rbm.w, _scale_w(rbm))

"""
    weight_norms(rbm::Union{CenteredRBM, StandardizedRBM})

Norms of the unstandardized weights attached to each hidden unit. For the norms of the
standardized weights, use `weight_norms(RBM(rbm))`.
"""
weight_norms(rbm::OffsetRBM) = weight_norms(RBM(rbm.visible, rbm.hidden, unstandardized_weights(rbm)))

function interaction_energy(rbm::OffsetRBM, v::AbstractArray, h::AbstractArray)
    return interaction_energy(RBM(rbm), standardize_v(rbm, v), standardize_h(rbm, h))
end

function inputs_h_from_v(rbm::OffsetRBM, v::AbstractArray)
    inputs = inputs_h_from_v(RBM(rbm), standardize_v(rbm, v))
    return _maybe_div(inputs, _scale_h(rbm))
end

function inputs_v_from_h(rbm::OffsetRBM, h::AbstractArray)
    inputs = inputs_v_from_h(RBM(rbm), standardize_h(rbm, h))
    return _maybe_div(inputs, _scale_v(rbm))
end

function free_energy(rbm::OffsetRBM, v::AbstractArray)
    E = energy(rbm.visible, v)
    inputs = inputs_h_from_v(rbm, v)
    F = -cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_h), inputs)
    return E + F - ΔE
end

function free_energy_h(rbm::OffsetRBM, h::AbstractArray)
    E = energy(rbm.hidden, h)
    inputs = inputs_v_from_h(rbm, h)
    F = -cgf(rbm.visible, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_v), inputs)
    return E + F - ΔE
end

function ∂interaction_energy(rbm::OffsetRBM, v::AbstractArray, h::AbstractArray; kwargs...)
    return ∂interaction_energy(RBM(rbm), standardize_v(rbm, v), standardize_h(rbm, h); kwargs...)
end

function log_pseudolikelihood(rbm::OffsetRBM, v::AbstractArray; kwargs...)
    return log_pseudolikelihood(_equivalent_rbm(rbm), v; kwargs...)
end

function ∂regularize!(
        ∂::∂RBM, offset_rbm::OffsetRBM;
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
        rbm = _equivalent_rbm(offset_rbm)
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

function regularization_penalty(rbm::OffsetRBM; regularize_unstandardized::Bool = true, kwargs...)
    return regularization_penalty(regularize_unstandardized ? _equivalent_rbm(rbm) : RBM(rbm); kwargs...)
end

"""
    zerosum!(rbm::Union{CenteredRBM, StandardizedRBM})

In-place version of `zerosum(rbm)`. Offsets (and scales) are not modified.
"""
function zerosum!(rbm::OffsetRBM)
    if rbm.visible isa _PottsLayers
        # Gauge move on the weights of the equivalent plain RBM, w̃ = w / (scale_v ⊗ scale_h):
        # subtract their mean over visible colors (scale_h cancels out of the w update).
        scale_v = _scale_v(rbm)
        ξ = mean(_maybe_div(rbm.w, scale_v); dims = 1)
        rbm.w .-= _maybe_mul(ξ, scale_v)
        zerosum!(rbm.visible.θ; dims = 1)
        # Compensate hidden fields. Unlike a plain RBM, the interaction involves
        # v - offset_v, so the color-sum of the visible offsets enters the shift.
        vdims = ntuple(identity, ndims(rbm.visible))
        Ov = sum(rbm.offset_v; dims = 1)
        Δθh = _maybe_div(reshape(sum(ξ .* (1 .- Ov); dims = vdims), size(rbm.hidden)), _scale_h(rbm))
        shift_fields!(rbm.hidden, Δθh)
    end
    if rbm.hidden isa _PottsLayers
        scale_h = _along_hidden(rbm, _scale_h(rbm))
        ζ = mean(_maybe_div(rbm.w, scale_h); dims = ndims(rbm.visible) + 1)
        rbm.w .-= _maybe_mul(ζ, scale_h)
        zerosum!(rbm.hidden.θ; dims = 1)
        hdims = ntuple(d -> d + ndims(rbm.visible), ndims(rbm.hidden))
        Oh = reshape(sum(rbm.offset_h; dims = 1), map(one, size(rbm.visible))..., 1, size(rbm.hidden)[2:end]...)
        Δθv = _maybe_div(reshape(sum(ζ .* (1 .- Oh); dims = hdims), size(rbm.visible)), _scale_v(rbm))
        shift_fields!(rbm.visible, Δθv)
    end
    return rbm
end

"""
    zerosum!(∂, rbm::Union{CenteredRBM, StandardizedRBM})

Projects the gradient so that it doesn't modify the zerosum gauge of the equivalent
plain `RBM` (see [`uncenter`](@ref), [`unstandardize`](@ref)), with offsets and scales
held fixed. The gauge condition on the weights reads `sum(w ./ scale_v; dims = 1) == 0`
over Potts colors (similarly for hidden Potts with `scale_h`), so the component removed
is the gauge direction `ξ .* scale_v`.
"""
function zerosum!(∂::∂RBM, rbm::OffsetRBM)
    if rbm.visible isa _PottsLayers
        zerosum!(∂.visible; dims = 2) # dim 1 of `par` is the (singleton) parameter type
        scale_v = _scale_v(rbm)
        ξ = mean(_maybe_div(∂.w, scale_v); dims = 1)
        ∂.w .-= _maybe_mul(ξ, scale_v)
    end
    if rbm.hidden isa _PottsLayers
        zerosum!(∂.hidden; dims = 2)
        scale_h = _along_hidden(rbm, _scale_h(rbm))
        ζ = mean(_maybe_div(∂.w, scale_h); dims = ndims(rbm.visible) + 1)
        ∂.w .-= _maybe_mul(ζ, scale_h)
    end
    return ∂
end
