@doc raw"""
    StandardizedRBM{V,H,W,Ov,Oh,Sv,Sh}

RBM with standardized layer activations. Like [`CenteredRBM`](@ref) it subtracts the
offsets `offset_v`, `offset_h` from the visible and hidden activations entering the
interaction, and additionally divides them by the scales `scale_v`, `scale_h`. The
energy is

```math
E(v,h) = E_v(v) + E_h(h) - \sum_{i\mu} w_{i\mu}
    \frac{v_i - \lambda_i}{\sigma_i} \frac{h_\mu - \lambda_\mu}{\sigma_\mu}
```

where ``\lambda`` are the offsets and ``\sigma`` the scales. A `CenteredRBM` is the
special case with unit scales. See <http://jmlr.org/papers/v17/14-237.html>.
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

RBM(rbm::StandardizedRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

standardize_v(rbm::StandardizedRBM, v::AbstractArray) = (v .- rbm.offset_v) ./ rbm.scale_v
standardize_h(rbm::StandardizedRBM, h::AbstractArray) = (h .- rbm.offset_h) ./ rbm.scale_h

function mirror(rbm::StandardizedRBM)
    _rbm = mirror(RBM(rbm))
    return StandardizedRBM(_rbm, rbm.offset_h, rbm.offset_v, rbm.scale_h, rbm.scale_v)
end

function rescale_hidden_activations!(rbm::StandardizedRBM)
    if rescale_activations!(rbm.hidden, rbm.scale_h)
        rbm.offset_h ./= rbm.scale_h
        rbm.scale_h ./= rbm.scale_h
        return true
    end
    return false
end

"""
    zerosum(rbm::StandardizedRBM)

Returns an equivalent `StandardizedRBM`, with the same offsets and scales, whose
equivalent unstandardized `RBM` (see [`unstandardize`](@ref)) is in the zerosum gauge.
Only affects Potts layers. If the `rbm` doesn't have `Potts` layers, does nothing.

Note that the gauge condition applies to the unstandardized parameters: the standardized
weights and fields need not sum to zero over Potts colors, because the interaction energy
involves the standardized `(v - offset_v) / scale_v`, for which sums over colors are not
constant when the offsets and scales vary across colors.
"""
function zerosum(rbm::StandardizedRBM)
    has_potts_layers(rbm) || return rbm
    plain = zerosum(unstandardize(rbm))
    return standardize(plain, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

"""
    delta_energy(rbm)

Compute the (constant) energy shift with respect to the equivalent normal RBM.
"""
delta_energy(rbm::RBM) = 0
function delta_energy(rbm::StandardizedRBM)
    v = Zeros(rbm.offset_v)
    h = Zeros(rbm.offset_h)
    return interaction_energy(rbm, v, h)
end

"""
    unstandardize(rbm)

Convert a `StandardizedRBM` back to an equivalent plain `RBM`.
Note: this does not enforce zerosum gauge; call `zerosum(unstandardize(rbm))` if needed.
"""
unstandardize(rbm::StandardizedRBM) = RBM(standardize(rbm))
unstandardize(rbm::RBM) = rbm

@doc raw"""
    standardize(rbm, offset_v = 0, offset_h = 0, scale_v = 1, scale_h = 1)

Constructs a `StandardizedRBM` equivalent to the given `rbm` (a plain `RBM` or another
`StandardizedRBM`), with the given offsets and scales. The energies assigned by the two
models differ by a constant amount, so the modeled distribution is unchanged.

This is the inverse operation of [`unstandardize`](@ref). To construct a
`StandardizedRBM` that simply adopts these offsets and scales *without* preserving the
distribution, call `StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)` instead.
"""
standardize(rbm::RBM) = StandardizedRBM(rbm)
function standardize(rbm::StandardizedRBM)
    offset_v = Zeros(rbm.offset_v)
    offset_h = Zeros(rbm.offset_h)
    scale_v = Ones(rbm.scale_v)
    scale_h = Ones(rbm.scale_h)
    return standardize(rbm, offset_v, offset_h, scale_v, scale_h)
end

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
    std_rbm = standardize(rbm)
    return standardize(std_rbm, offset_v, offset_h, scale_v, scale_h)
end

function standardize_visible(std_rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(std_rbm.visible) == size(offset_v) == size(scale_v)

    cv = scale_v ./ std_rbm.scale_v
    Δθ = inputs_h_from_v(std_rbm, offset_v)

    hid = shift_fields(std_rbm.hidden, Δθ)
    w = std_rbm.w .* cv
    rbm = RBM(std_rbm.visible, hid, w)

    return StandardizedRBM(rbm, offset_v, std_rbm.offset_h, scale_v, std_rbm.scale_h)
end

function standardize_hidden(std_rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(std_rbm.hidden) == size(offset_h) == size(scale_h)

    ch = reshape(scale_h ./ std_rbm.scale_h, map(one, size(std_rbm.visible))..., size(std_rbm.hidden)...)
    Δθ = inputs_v_from_h(std_rbm, offset_h)

    vis = shift_fields(std_rbm.visible, Δθ)
    w = std_rbm.w .* ch
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
    standardize!(rbm::StandardizedRBM, offset_v, offset_h, scale_v, scale_h)

Transforms the offsets and scales of `rbm` in place. The transformed model is equivalent
to the original one (energies differ by a constant). In-place analogue of
[`standardize`](@ref).
"""
function standardize!(rbm::StandardizedRBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    standardize_visible!(rbm, offset_v, scale_v)
    standardize_hidden!(rbm, offset_h, scale_h)
    return rbm
end

function standardize_visible!(rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)

    cv = scale_v ./ rbm.scale_v
    Δθ = inputs_h_from_v(rbm, offset_v)

    shift_fields!(rbm.hidden, Δθ)
    rbm.w .= rbm.w .* cv
    rbm.offset_v .= offset_v
    rbm.scale_v .= scale_v

    return rbm
end

function standardize_hidden!(rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)

    ch = reshape(scale_h ./ rbm.scale_h, (map(one, size(rbm.visible))..., size(rbm.hidden)...))
    Δθ = inputs_v_from_h(rbm, offset_h)

    shift_fields!(rbm.visible, Δθ)
    rbm.w .= rbm.w .* ch
    rbm.offset_h .= offset_h
    rbm.scale_h .= scale_h

    return rbm
end

function standardize_visible_from_data!(rbm::StandardizedRBM, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    μ = batchmean(rbm.visible, data; wts)
    ν = batchvar(rbm.visible, data; wts, mean = μ)
    scale = sqrt.(ν .+ ϵ)
    # Centered constant coordinates are identically zero, so their scale is arbitrary.
    # Use the neutral unit scale to avoid dividing by zero during standardization.
    @. scale = ifelse(iszero(scale), one(scale), scale)
    return standardize_visible!(rbm, μ, scale)
end

function standardize_hidden_from_inputs!(rbm::StandardizedRBM, inputs::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    μ, ν = total_meanvar_from_inputs(rbm.hidden, inputs; wts)
    offset_h = (1 - damping) .* rbm.offset_h + damping .* μ
    scale_h = sqrt.((1 - damping) .* rbm.scale_h .^ 2 + damping .* (ν .+ ϵ))
    return standardize_hidden!(rbm, offset_h, scale_h)
end

function standardize_hidden_from_v!(rbm::StandardizedRBM, v::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    inputs = inputs_h_from_v(rbm, v)
    return standardize_hidden_from_inputs!(rbm, inputs; damping, wts, ϵ)
end

unstandardized_weights(rbm::StandardizedRBM) = rbm.w ./ _scale_w(rbm)

function potts_to_gumbel(rbm::StandardizedRBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

function gumbel_to_potts(rbm::StandardizedRBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

function pcd!(
        rbm::StandardizedRBM,
        data::AbstractArray;

        batchsize::Int = 1,
        shuffle::Bool = true,

        iters::Int = 1, # number of gradient updates
        wts::Union{AbstractVector, Nothing} = nothing, # data weights

        steps::Int = 1,
        vm::Union{AbstractArray, Nothing} = nothing,

        moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

        # regularization
        l2_fields::Real = 0, # visible fields L2 regularization
        l1_weights::Real = 0, # weights L1 regularization
        l2_weights::Real = 0, # weights L2 regularization
        l2l1_weights::Real = 0, # weights L2/L1 regularization

        # "pseudocount" for estimating variances of v and h and damping
        damping::Real = 1 // 100, ϵv::Real = 0, ϵh::Real = 0,

        # whether regularization applies to unstandardized model parameters (default),
        # or to the parameters of the standardized model
        regularize_unstandardized::Bool = true,

        # optimiser
        optim::AbstractRule = Adam(),
        ps = nothing,
        state = nothing,

        # Absorb the scale_h into the hidden unit activation (for hidden units with scale parameter).
        # Results in hidden units with var(h) ~ 1.
        rescale_hidden::Bool = true,

        zerosum::Bool = true, # zerosum gauge for Potts layers

        # called for every gradient update
        callback = Returns(nothing)
    )
    @assert 0 ≤ damping ≤ 1
    _validate_layer_parameters(rbm)
    isnothing(vm) && (vm = _default_fantasy_chains(rbm, batchsize))
    return _train!(
        rbm, data;
        batchsize, iters, wts, moments, optim, ps, state, shuffle,
        l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum, callback,
        regularize = (; regularize_unstandardized),
        setup! = (data, wts) -> begin
            standardize_visible_from_data!(rbm, data; wts, ϵ = ϵv)
            zerosum && zerosum!(rbm)
        end,
        negative_phase = vd -> _pcd_negative_phase(rbm, vm, steps),
        post_update! = (vd, wd, ∂d) -> begin
            # update standardization
            standardize_hidden_from_v!(rbm, vd; wts = wd, damping, ϵ = ϵh)
            zerosum && zerosum!(rbm)
            rescale_hidden && rescale_hidden_activations!(rbm)
        end,
    )
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

function log_partition(rbm::StandardizedRBM)
    v = ChainRulesCore.ignore_derivatives() do
        collect_states(rbm.visible)
    end
    return logsumexp(-free_energy(rbm, v))
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

"""
    weight_norms(std_rbm::StandardizedRBM)

Computes the norms of the unstandardized weights for each hidden unit. If you want the norms
of the standardized weights, use `weight_norms(RBM(std_rbm))`.
"""
function weight_norms(rbm::StandardizedRBM)
    w = unstandardized_weights(rbm)
    w2 = sum(abs2, w; dims = 1:ndims(rbm.visible))
    return reshape(sqrt.(w2), size(rbm.hidden))
end
