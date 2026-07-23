#= Shared implementations for `CenteredRBM` and `StandardizedRBM`.

Both models subtract offsets from the layer activations entering the interaction energy;
`StandardizedRBM` additionally divides by scales. A `CenteredRBM` therefore behaves like a
`StandardizedRBM` with unit scales, which the `_scale_v` / `_scale_h` accessors expose as
`Ones` so that each method can be written once. The `_maybe_div` / `_maybe_mul` helpers
skip the scaling no-op in the `CenteredRBM` case, keeping its hot paths free of spurious
divisions by one. =#

const OffsetRBM = Union{CenteredRBM, StandardizedRBM}

standardize_v(rbm::CenteredRBM, v::AbstractArray) = v .- rbm.offset_v
standardize_h(rbm::CenteredRBM, h::AbstractArray) = h .- rbm.offset_h

_scale_v(rbm::StandardizedRBM) = rbm.scale_v
_scale_h(rbm::StandardizedRBM) = rbm.scale_h
_scale_v(rbm::CenteredRBM) = Ones{eltype(rbm.w)}(size(rbm.visible))
_scale_h(rbm::CenteredRBM) = Ones{eltype(rbm.w)}(size(rbm.hidden))

_maybe_div(x::AbstractArray, ::Ones) = x
_maybe_div(x::AbstractArray, s::AbstractArray) = x ./ s
_maybe_mul(x::AbstractArray, ::Ones) = x
_maybe_mul(x::AbstractArray, s::AbstractArray) = x .* s

# scales of the weights of the equivalent plain RBM, shaped like `rbm.w`
function _scale_w(rbm::StandardizedRBM)
    cv = reshape(rbm.scale_v, size(rbm.visible)..., map(one, size(rbm.hidden))...)
    ch = reshape(rbm.scale_h, map(one, size(rbm.visible))..., size(rbm.hidden)...)
    return cv .* ch
end
_scale_w(rbm::CenteredRBM) = Ones{eltype(rbm.w)}(size(rbm.w))

# equivalent plain `RBM` modeling the same distribution
_equivalent_rbm(rbm::CenteredRBM) = uncenter(rbm)
_equivalent_rbm(rbm::StandardizedRBM) = unstandardize(rbm)

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

function ∂interaction_energy(rbm::OffsetRBM, v::AbstractArray, h::AbstractArray; wts = nothing)
    return ∂interaction_energy(RBM(rbm), standardize_v(rbm, v), standardize_h(rbm, h); wts)
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
    if regularize_unstandardized
        # regularization applies to the parameters of the equivalent plain RBM
        rbm = _equivalent_rbm(offset_rbm)
        offset_h = reshape(offset_rbm.offset_h, map(one, size(offset_rbm.offset_v))..., size(offset_rbm.offset_h)...)
        scale_w = _scale_w(offset_rbm)

        if !iszero(l2_fields)
            visible_reg = ∂regularize_fields(rbm.visible; l2_fields)
            ∂.visible .+= visible_reg
            ∂regularize_add_visible_offset!(∂, visible_reg, offset_h, scale_w, offset_rbm.visible)
        end
        if !iszero(l1_weights)
            ∂.w .+= _maybe_div(l1_weights * sign.(rbm.w), scale_w)
        end
        if !iszero(l2_weights)
            ∂.w .+= _maybe_div(l2_weights * rbm.w, scale_w)
        end
        if !iszero(l2l1_weights)
            dims = ntuple(identity, ndims(offset_rbm.visible))
            ∂.w .+= _maybe_div(l2l1_weights * sign.(rbm.w) .* mean(abs, rbm.w; dims), scale_w)
        end
    else
        # regularization applies directly to the offset model's parameters
        ∂regularize!(∂, RBM(offset_rbm); l2_fields, l1_weights, l2_weights, l2l1_weights)
    end
    zerosum && zerosum!(∂, offset_rbm)
    return ∂
end

function ∂regularize_add_visible_offset!(∂::∂RBM, visible_regularization::AbstractArray, offset_h::AbstractArray, scale_w::AbstractArray, ::dReLU)
    return ∂.w .-= _maybe_div((visible_regularization[1, ..] + visible_regularization[2, ..]) .* offset_h, scale_w)
end

function ∂regularize_add_visible_offset!(∂::∂RBM, visible_regularization::AbstractArray, offset_h::AbstractArray, scale_w::AbstractArray, ::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, xReLU, pReLU, nsReLU})
    return ∂.w .-= _maybe_div(visible_regularization[1, ..] .* offset_h, scale_w)
end

function regularization_penalty(
        rbm::OffsetRBM; regularize_unstandardized::Bool = true,
        l1_weights::Real = 0, l2_weights::Real = 0, l2l1_weights::Real = 0, l2_fields::Real = 0,
    )
    if regularize_unstandardized
        return regularization_penalty(_equivalent_rbm(rbm); l1_weights, l2_weights, l2l1_weights, l2_fields)
    else
        return regularization_penalty(RBM(rbm); l1_weights, l2_weights, l2l1_weights, l2_fields)
    end
end

"""
    zerosum!(rbm::Union{CenteredRBM, StandardizedRBM})

In-place version of `zerosum(rbm)`. Offsets (and scales) are not modified.
"""
function zerosum!(rbm::OffsetRBM)
    if rbm.visible isa Union{Potts, PottsGumbel}
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
    if rbm.hidden isa Union{Potts, PottsGumbel}
        scale_h = reshape(_scale_h(rbm), map(one, size(rbm.visible))..., size(rbm.hidden)...)
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
held fixed.

The gauge condition applies to the parameters of the equivalent plain RBM: for the
weights it reads `sum(w ./ scale_v; dims = 1) == 0` over Potts colors (and similarly for
hidden Potts with `scale_h`), so the gradient component removed here is the corresponding
gauge direction `ξ .* scale_v`. For a `CenteredRBM` the scales are one and these
conditions coincide with the plain `RBM` ones.
"""
function zerosum!(∂::∂RBM, rbm::OffsetRBM)
    if rbm.visible isa Union{Potts, PottsGumbel}
        zerosum!(∂.visible; dims = 2) # dim 1 of `par` is the (singleton) parameter type
        scale_v = _scale_v(rbm)
        ξ = mean(_maybe_div(∂.w, scale_v); dims = 1)
        ∂.w .-= _maybe_mul(ξ, scale_v)
    end
    if rbm.hidden isa Union{Potts, PottsGumbel}
        zerosum!(∂.hidden; dims = 2)
        scale_h = reshape(_scale_h(rbm), map(one, size(rbm.visible))..., size(rbm.hidden)...)
        ζ = mean(_maybe_div(∂.w, scale_h); dims = ndims(rbm.visible) + 1)
        ∂.w .-= _maybe_mul(ζ, scale_h)
    end
    return ∂
end
