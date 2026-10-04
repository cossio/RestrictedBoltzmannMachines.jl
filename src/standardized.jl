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

standardize_v(rbm::StandardizedRBM, v::AbstractArray) = (v .- rbm.offset_v) ./ rbm.scale_v
standardize_h(rbm::StandardizedRBM, h::AbstractArray) = (h .- rbm.offset_h) ./ rbm.scale_h

function mirror(rbm::StandardizedRBM)
    _rbm = mirror(RBM(rbm))
    return StandardizedRBM(_rbm, rbm.offset_h, rbm.offset_v, rbm.scale_h, rbm.scale_v)
end

"""
    rescale_hidden_activations!(rbm::StandardizedRBM)

Absorbs `scale_h` into the hidden layer if it has a scale parameter, returning `true`
if this was done. The modified RBM is equivalent to the original one.
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
    standardize(rbm, offset_v = 0, offset_h = 0, scale_v = 1, scale_h = 1)

Constructs a `StandardizedRBM` equivalent to the given `rbm` (a plain `RBM` or another
`StandardizedRBM`), with the given offsets and scales. The energies assigned by the two
models differ by a constant amount, so the modeled distribution is unchanged.

This is the inverse operation of [`unstandardize`](@ref). To construct a
`StandardizedRBM` that simply adopts these offsets and scales *without* preserving the
distribution, call `StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)` instead.
"""
standardize(rbm::RBM) = StandardizedRBM(rbm)
standardize(rbm::StandardizedRBM) = standardize(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h), Ones(rbm.scale_v), Ones(rbm.scale_h))

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

    ch = _along_hidden(std_rbm, scale_h ./ std_rbm.scale_h)
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

    ch = _along_hidden(rbm, scale_h ./ rbm.scale_h)
    Δθ = inputs_v_from_h(rbm, offset_h)

    shift_fields!(rbm.visible, Δθ)
    rbm.w .= rbm.w .* ch
    rbm.offset_h .= offset_h
    rbm.scale_h .= scale_h

    return rbm
end

"""
    standardize_visible_from_data!(rbm::StandardizedRBM, data; [wts], ϵ = 0)

Sets the visible offsets and scales to the mean and standard deviation of `data`.
The model is unchanged (energies differ by a constant).
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
        wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs), damping::Real = 0, ϵ::Real = 0
    )
    μ, ν = total_meanvar_from_inputs(rbm.hidden, inputs; wts)
    offset_h = (1 - damping) .* rbm.offset_h + damping .* μ
    scale_h = sqrt.((1 - damping) .* rbm.scale_h .^ 2 + damping .* (ν .+ ϵ))
    return standardize_hidden!(rbm, offset_h, scale_h)
end

"""
    standardize_hidden_from_v!(rbm::StandardizedRBM, v; [wts], damping = 0, ϵ = 0)

Sets the hidden offsets and scales to the mean and standard deviation of hidden unit
activations conditioned on `v`. The model is unchanged (energies differ by a constant).
"""
function standardize_hidden_from_v!(
        rbm::StandardizedRBM, v::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, v), damping::Real = 0, ϵ::Real = 0
    )
    inputs = inputs_h_from_v(rbm, v)
    return standardize_hidden_from_inputs!(rbm, inputs; damping, wts, ϵ)
end

"""
    pcd!(rbm::StandardizedRBM, data; damping = 1 // 100, ϵv = 0, ϵh = 0,
         regularize_unstandardized = true, rescale_hidden = true, kwargs...)

[`pcd!`](@ref) for a `StandardizedRBM`, with the same keywords as for a plain `RBM` except
`rescale`. The visible offsets and scales are set from `data` before training, and after
every update the hidden offsets and scales move towards the minibatch conditional
statistics by a fraction `damping`; `ϵv`, `ϵh` are pseudocounts added to the variances.
If `rescale_hidden`, `scale_h` is absorbed into hidden units that have a scale parameter,
so that `var(h) ≈ 1`. Regularization applies to the equivalent plain `RBM` if
`regularize_unstandardized`, otherwise to the standardized parameters.
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
        rescale_hidden::Bool = true, # absorb scale_h into hidden units with a scale parameter, so var(h) ~ 1
        callback = Returns(nothing),
        vm::AbstractArray = _default_fantasy_chains(rbm, min(batchsize, size(data)[end])),
        shuffle::Bool = true,
        ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
        state = setup(optim, ps),
    )
    @assert 0 ≤ damping ≤ 1
    wts_mean, batchsize = _pcd_check_args(rbm, data, wts, batchsize)

    standardize_visible_from_data!(rbm, data; wts, ϵ = ϵv)
    zerosum && zerosum!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        state, ps, ∂ = _pcd_step!(
            rbm, ps, state, vd, wd, vm, wts_mean;
            steps, moments, l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum,
            regularize_unstandardized
        )

        # update standardization
        standardize_hidden_from_v!(rbm, vd; wts = wd, damping, ϵ = ϵh)
        # zerosum! first because absorbing scale_h preserves the zero-sum gauge
        zerosum && zerosum!(rbm)
        rescale_hidden && rescale_hidden_activations!(rbm)

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
