@doc raw"""
    StandardizedRBM{V,H,W,Ov,Oh,Sv,Sh}

RBM with standardized layer activations. It subtracts the offsets `offset_v`, `offset_h`
from the visible and hidden activations entering the interaction, and divides them by the
scales `scale_v`, `scale_h`. The energy is

```math
E(v,h) = E_v(v) + E_h(h) - \sum_{i\mu} w_{i\mu}
    \frac{v_i - \lambda_i}{\sigma_i} \frac{h_\mu - \lambda_\mu}{\sigma_\mu}
```

where ``\lambda`` are the offsets and ``\sigma`` the scales. A [`CenteredRBM`](@ref) is
the special case whose scales are fixed to one.
See <http://jmlr.org/papers/v17/14-237.html>.
"""
struct StandardizedRBM{V, H, W, Ov, Oh, Sv, Sh}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    scale_v::Sv
    scale_h::Sh
    function StandardizedRBM(
            visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
            offset_v::AbstractArray, offset_h::AbstractArray,
            scale_v::AbstractArray, scale_h::AbstractArray
        )
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(visible) == size(offset_v) == size(scale_v)
        @assert size(hidden) == size(offset_h) == size(scale_h)
        V, H, W = typeof(visible), typeof(hidden), typeof(w)
        Ov, Oh, Sv, Sh = typeof(offset_v), typeof(offset_h), typeof(scale_v), typeof(scale_h)
        return new{V, H, W, Ov, Oh, Sv, Sh}(visible, hidden, w, offset_v, offset_h, scale_v, scale_h)
    end
end

"""
    StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)

Creates a standardized RBM, with offsets `offset_v`, `offset_h` and scales `scale_v`,
`scale_h`. The resulting model is *not* equivalent to the original `rbm`, unless the
offsets are zero and the scales are one. To construct an equivalent model instead, use
[`standardize`](@ref).
"""
function StandardizedRBM(
        rbm::RBM,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
    return StandardizedRBM(rbm.visible, rbm.hidden, rbm.w, offset_v, offset_h, scale_v, scale_h)
end

"""
    StandardizedRBM(rbm)

Creates a standardized RBM from `rbm`, with offsets initialized to zero and scales to
one (so the constructed model is equivalent to `rbm`).
"""
function StandardizedRBM(rbm::RBM)
    offset_v = zeros_like(rbm.w, size(rbm.visible))
    offset_h = zeros_like(rbm.w, size(rbm.hidden))
    scale_v = ones_like(rbm.w, size(rbm.visible))
    scale_h = ones_like(rbm.w, size(rbm.hidden))
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

standardize_v(rbm::StandardizedRBM, v::AbstractArray) = _maybe_div(_maybe_sub(v, rbm.offset_v), rbm.scale_v)
standardize_h(rbm::StandardizedRBM, h::AbstractArray) = _maybe_div(_maybe_sub(h, rbm.offset_h), rbm.scale_h)

"""
    RBM(rbm::StandardizedRBM)

Plain `RBM` sharing the layers and weights of `rbm` but ignoring its offsets and scales, so
it is *not* equivalent to `rbm`. For an equivalent model use [`unstandardize`](@ref).
"""
RBM(rbm::StandardizedRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

# scales of the weights of the equivalent plain RBM, shaped like `rbm.w` (lazy for a CenteredRBM)
_scale_w(rbm::StandardizedRBM) = _along_visible(rbm, rbm.scale_v) .* _along_hidden(rbm, rbm.scale_h)

"""
    delta_energy(rbm)

The constant energy shift of `rbm` with respect to its equivalent plain `RBM`.
"""
delta_energy(rbm::RBM) = 0
delta_energy(rbm::StandardizedRBM) = interaction_energy(rbm, Falses(size(rbm.visible)), Falses(size(rbm.hidden)))

potts_to_gumbel(rbm::StandardizedRBM) =
    StandardizedRBM(potts_to_gumbel(RBM(rbm)), rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
gumbel_to_potts(rbm::StandardizedRBM) =
    StandardizedRBM(gumbel_to_potts(RBM(rbm)), rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)

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

function mirror(rbm::StandardizedRBM)
    _rbm = mirror(RBM(rbm))
    return StandardizedRBM(_rbm, rbm.offset_h, rbm.offset_v, rbm.scale_h, rbm.scale_v)
end

"""
    rescale_hidden_activations!(rbm::StandardizedRBM)

Fixes the scale gauge of hidden units that have a scale parameter, returning `true` if
they do. A `StandardizedRBM` absorbs `scale_h` into the hidden layer, so that `scale_h`
becomes one and `var(h) ≈ 1`. The scales of a [`CenteredRBM`](@ref) are fixed to one, so it
normalizes the weights of each hidden unit instead (see [`rescale_weights!`](@ref)). The
modified RBM is equivalent to the original one.
"""
function rescale_hidden_activations!(rbm::StandardizedRBM)
    return rescale_hidden!(rbm, copy(rbm.scale_h))
end

"""
    unstandardize(rbm)

Convert a `StandardizedRBM` back to an equivalent plain `RBM`.
Note: this does not enforce zerosum gauge; call `zerosum(unstandardize(rbm))` if needed.
"""
unstandardize(rbm::StandardizedRBM) = RBM(standardize(rbm))
unstandardize(rbm::RBM) = rbm

@doc raw"""
    standardize(rbm)
    standardize(rbm, offset_v, offset_h)
    standardize(rbm, offset_v, offset_h, scale_v, scale_h)

Constructs a `StandardizedRBM` equivalent to the given `rbm` (a plain `RBM` or another
`StandardizedRBM`), with the given offsets and scales. The energies assigned by the two
models differ by a constant amount, so the modeled distribution is unchanged.

Omitted offsets are zero. Omitted scales are one, except that the three-argument form
keeps the scales of `rbm`: those of a `StandardizedRBM`, or unit scales for a plain `RBM`,
which gives a [`CenteredRBM`](@ref).

This is the inverse operation of [`unstandardize`](@ref). To construct a
`StandardizedRBM` that simply adopts these offsets and scales *without* preserving the
distribution, call `StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)` instead.
"""
standardize(rbm::RBM) = StandardizedRBM(rbm)
standardize(rbm::StandardizedRBM) = standardize(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h), Ones(rbm.scale_v), Ones(rbm.scale_h))
standardize(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray) = standardize(CenteredRBM(rbm), offset_v, offset_h)
standardize(rbm::StandardizedRBM, offset_v::AbstractArray, offset_h::AbstractArray) =
    standardize(rbm, offset_v, offset_h, rbm.scale_v, rbm.scale_h)

function standardize(
        rbm::StandardizedRBM,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    std_rbm = standardize_visible(rbm, offset_v, scale_v)
    return standardize_hidden(std_rbm, offset_h, scale_h)
end

function standardize(
        rbm::RBM,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
    return standardize(CenteredRBM(rbm), offset_v, offset_h, scale_v, scale_h)
end

function standardize_visible(std_rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(std_rbm.visible) == size(offset_v) == size(scale_v)

    cv = _maybe_div(scale_v, std_rbm.scale_v)
    Δθ = inputs_h_from_v(std_rbm, offset_v)

    hid = shift_fields(std_rbm.hidden, Δθ)
    w = _maybe_mul(std_rbm.w, cv)
    rbm = RBM(std_rbm.visible, hid, w)

    return StandardizedRBM(rbm, offset_v, std_rbm.offset_h, scale_v, std_rbm.scale_h)
end

function standardize_hidden(std_rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(std_rbm.hidden) == size(offset_h) == size(scale_h)

    ch = _along_hidden(std_rbm, _maybe_div(scale_h, std_rbm.scale_h))
    Δθ = inputs_v_from_h(std_rbm, offset_h)

    vis = shift_fields(std_rbm.visible, Δθ)
    w = _maybe_mul(std_rbm.w, ch)
    rbm = RBM(vis, std_rbm.hidden, w)

    return StandardizedRBM(rbm, std_rbm.offset_v, offset_h, std_rbm.scale_v, scale_h)
end

standardize_visible(rbm::RBM, offset_v::AbstractArray, scale_v::AbstractArray) = standardize_visible(standardize(rbm), offset_v, scale_v)
standardize_hidden(rbm::RBM, offset_h::AbstractArray, scale_h::AbstractArray) = standardize_hidden(standardize(rbm), offset_h, scale_h)

standardize_visible(rbm::StandardizedRBM) = standardize_visible(rbm, zero(rbm.offset_v), one.(rbm.scale_v))
standardize_hidden(rbm::StandardizedRBM) = standardize_hidden(rbm, zero(rbm.offset_h), one.(rbm.scale_h))
standardize_visible(rbm::RBM) = standardize(rbm)
standardize_hidden(rbm::RBM) = standardize(rbm)

"""
    standardize!(rbm::StandardizedRBM)
    standardize!(rbm::StandardizedRBM, offset_v, offset_h)
    standardize!(rbm::StandardizedRBM, offset_v, offset_h, scale_v, scale_h)

Transforms the offsets and scales of `rbm` in place. The transformed model is equivalent
to the original one (energies differ by a constant). In-place analogue of
[`standardize`](@ref): omitted offsets are zero, and omitted scales are one, or are kept
by the three-argument form. The scales of a [`CenteredRBM`](@ref) are fixed to one, so
other scales throw an error; use [`standardize`](@ref) instead.
"""
function standardize!(rbm::StandardizedRBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    standardize_visible!(rbm, offset_v, scale_v)
    standardize_hidden!(rbm, offset_h, scale_h)
    return rbm
end

standardize!(rbm::StandardizedRBM, offset_v::AbstractArray, offset_h::AbstractArray) =
    standardize!(rbm, offset_v, offset_h, rbm.scale_v, rbm.scale_h)
standardize!(rbm::StandardizedRBM) =
    standardize!(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h), Ones(rbm.scale_v), Ones(rbm.scale_h))

function standardize_visible!(rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)

    cv = _maybe_div(scale_v, rbm.scale_v)
    Δθ = inputs_h_from_v(rbm, offset_v)

    rbm.scale_v .= scale_v # first: lazy unit scales (`Ones`) throw unless set to one
    shift_fields!(rbm.hidden, Δθ)
    _maybe_mul!(rbm.w, cv)
    rbm.offset_v .= offset_v

    return rbm
end

function standardize_hidden!(rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)

    ch = _along_hidden(rbm, _maybe_div(scale_h, rbm.scale_h))
    Δθ = inputs_v_from_h(rbm, offset_h)

    rbm.scale_h .= scale_h # first: lazy unit scales (`Ones`) throw unless set to one
    shift_fields!(rbm.visible, Δθ)
    _maybe_mul!(rbm.w, ch)
    rbm.offset_h .= offset_h

    return rbm
end

"""
    standardize_visible_from_data!(rbm::StandardizedRBM, data; [wts], ϵ = 0)

Sets the visible offsets and scales to the mean and standard deviation of `data`; the
scales of a [`CenteredRBM`](@ref) stay fixed to one. The model is unchanged (energies
differ by a constant).
"""
function standardize_visible_from_data!(
        rbm::StandardizedRBM, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
    )
    μ = batchmean(rbm.visible, data; wts)
    ν = batchvar(rbm.visible, data; wts, mean = μ)
    scale = sqrt.(ν .+ ϵ)
    # Centered constant coordinates are identically zero, so their scale is arbitrary.
    # Use the neutral unit scale to avoid dividing by zero during standardization.
    @. scale = ifelse(iszero(scale), one(scale), scale)
    return standardize_visible!(rbm, μ, scale)
end

function standardize_hidden_from_inputs!(
        rbm::StandardizedRBM, inputs::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs), damping::Real = 1, ϵ::Real = 0
    )
    μ, ν = total_meanvar_from_inputs(rbm.hidden, inputs; wts)
    offset_h = (1 - damping) .* rbm.offset_h + damping .* μ
    scale_h = sqrt.((1 - damping) .* rbm.scale_h .^ 2 + damping .* (ν .+ ϵ))
    return standardize_hidden!(rbm, offset_h, scale_h)
end

"""
    standardize_hidden_from_v!(rbm::StandardizedRBM, v; [wts], damping = 1, ϵ = 0)

Moves the hidden offsets and scales towards the mean and standard deviation of hidden
unit activations conditioned on `v`, by a fraction `damping` (`1` sets them); the scales of
a [`CenteredRBM`](@ref) stay fixed to one. The model is unchanged (energies differ by a
constant).
"""
function standardize_hidden_from_v!(
        rbm::StandardizedRBM, v::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, v), damping::Real = 1, ϵ::Real = 0
    )
    inputs = inputs_h_from_v(rbm, v)
    return standardize_hidden_from_inputs!(rbm, inputs; damping, wts, ϵ)
end

"""
    initialize!(rbm::StandardizedRBM, [data]; ϵ = 1e-6)

Initializes `rbm` to the standardized form of the plain `RBM` that
[`initialize!`](@ref) would produce from its layers and weights: the offsets and scales
are set from `data` (see [`standardize_visible_from_data!`](@ref) and
[`standardize_hidden_from_v!`](@ref)), or to zero and one if `data` is omitted. The
previous offsets and scales are discarded.
"""
function initialize!(rbm::StandardizedRBM; ϵ::Real = 1.0e-6)
    standardize!(rbm) # zero offsets and unit scales; the parameters are overwritten next
    initialize!(RBM(rbm); ϵ)
    return rbm
end

function initialize!(
        rbm::StandardizedRBM, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data)
    )
    standardize!(rbm) # zero offsets and unit scales; the parameters are overwritten next
    initialize!(RBM(rbm), data; ϵ, wts)
    standardize_visible_from_data!(rbm, data; wts)
    standardize_hidden_from_v!(rbm, data; wts)
    return rbm
end

"""
    pcd!(rbm::StandardizedRBM, data; damping = 1 // 100, ϵv = 0, ϵh = 0,
         regularize_unstandardized = true, kwargs...)

[`pcd!`](@ref) for a `StandardizedRBM` (including a [`CenteredRBM`](@ref)), with the same
keywords as for a plain `RBM`. The offsets and scales of both layers are set from `data`
before training, and after every update the hidden ones move towards the minibatch
conditional statistics by a fraction `damping`; `ϵv`, `ϵh` are pseudocounts added to the
variances. The scales of a `CenteredRBM` stay fixed to one. If `rescale`, the scale gauge
of the hidden units is fixed by [`rescale_hidden_activations!`](@ref). Regularization
applies to the equivalent plain `RBM` if `regularize_unstandardized`, otherwise to the
standardized parameters.
"""
function pcd!(
        rbm::StandardizedRBM,
        data::AbstractArray;
        batchsize::Int = 1,
        iters::Int = 1,
        wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data),
        steps::Int = 1,
        optim::AbstractRule = Adam(),
        moments = moments_from_samples(rbm.visible, data; wts),
        damping::Real = 1 // 100, # of the hidden standardization updates
        ϵv::Real = 0, ϵh::Real = 0, # pseudocounts for the visible and hidden variances
        regularize_unstandardized::Bool = true, # regularize the equivalent plain RBM, or this one
        l2_fields::Real = 0,
        l1_weights::Real = 0,
        l2_weights::Real = 0,
        l2l1_weights::Real = 0,
        zerosum::Bool = true,
        rescale::Bool = true,
        callback = Returns(nothing),
        vm::AbstractArray = _default_fantasy_chains(rbm, min(batchsize, size(data)[end])),
        ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
        state = setup(optim, ps),
    )
    @assert 0 ≤ damping ≤ 1
    wts_mean, batchsize = _pcd_check_args(rbm, data, wts, batchsize)

    standardize_visible_from_data!(rbm, data; wts, ϵ = ϵv)
    standardize_hidden_from_v!(rbm, data; wts, ϵ = ϵh)
    # zerosum! first because rescaling preserves the zero-sum gauge
    zerosum && zerosum!(rbm)
    rescale && rescale_hidden_activations!(rbm)

    for iter in 1:iters
        idx = sample(1:size(data)[end], batchsize; replace = false)
        vd, wd = data[.., idx], wts[idx]
        state, ps, ∂ = _pcd_step!(
            rbm, ps, state, vd, wd, vm, wts_mean;
            steps, moments, l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum,
            regularize_unstandardized
        )

        standardize_hidden_from_v!(rbm, vd; wts = wd, damping, ϵ = ϵh)
        zerosum && zerosum!(rbm)
        rescale && rescale_hidden_activations!(rbm)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)
    end
    return state, ps
end

"""
    BinaryStandardizedRBM(a, b, w, offset_v, offset_h, scale_v, scale_h)
    BinaryStandardizedRBM(a, b, w)

Construct a standardized RBM with `Binary` visible and hidden layers. With the short
form the offsets are zero and the scales one (equivalent to the plain `BinaryRBM`).
"""
function BinaryStandardizedRBM(
        a::AbstractArray, b::AbstractArray, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
    rbm = BinaryRBM(a, b, w)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

function BinaryStandardizedRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = BinaryRBM(a, b, w)
    return standardize(rbm)
end

"""
    SpinStandardizedRBM(a, b, w, offset_v, offset_h, scale_v, scale_h)
    SpinStandardizedRBM(a, b, w)

Construct a standardized RBM with `Spin` visible and hidden layers. With the short form
the offsets are zero and the scales one (equivalent to the plain `SpinRBM`).
"""
function SpinStandardizedRBM(
        a::AbstractArray, b::AbstractArray, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
    rbm = SpinRBM(a, b, w)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

function SpinStandardizedRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = SpinRBM(a, b, w)
    return standardize(rbm)
end

"""
    rescale_hidden!(rbm::StandardizedRBM, λ::AbstractArray)

Rescale hidden unit activities by `λ`, which should be an array of the same size as the hidden units.
This assumes the hidden units have a scale parameter, otherwise it does nothing and returns `false`.
"""
function rescale_hidden!(rbm::StandardizedRBM, λ::AbstractArray)
    @assert size(rbm.hidden) == size(λ)
    if rescale_activations!(rbm.hidden, λ)
        rbm.scale_h ./= λ
        rbm.offset_h ./= λ
        return true
    end
    return false
end
