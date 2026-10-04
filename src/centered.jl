"""
    CenteredRBM{V,H,W,Ov,Oh}

A [`StandardizedRBM`](@ref) whose scales are fixed to one, stored lazily (and immutably) as
`FillArrays.Trues`, so in-place updates change only its offsets (see [`center!`](@ref)).
See <http://jmlr.org/papers/v17/14-237.html>.
"""
const CenteredRBM{V, H, W, Ov, Oh} = StandardizedRBM{V, H, W, Ov, Oh, <:Trues, <:Trues}

"""
    CenteredRBM(rbm, λv, λh)

Creates a centered RBM, with offsets `λv` (visible) and `λh` (hidden).
See <http://jmlr.org/papers/v17/14-237.html> for details.
The resulting model is *not* equivalent to the original `rbm`, unless `λv = 0` and `λh = 0`.
"""
function CenteredRBM(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    return StandardizedRBM(rbm, offset_v, offset_h, Trues(size(rbm.visible)), Trues(size(rbm.hidden)))
end

function CenteredRBM(
        visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray
    )
    return CenteredRBM(RBM(visible, hidden, w), offset_v, offset_h)
end

"""
    CenteredRBM(rbm)
    CenteredRBM(visible, hidden, w)

Creates a centered RBM, with offsets initialized to zero.
"""
CenteredRBM(rbm::RBM) = CenteredRBM(rbm, zeros_like(rbm.w, size(rbm.visible)), zeros_like(rbm.w, size(rbm.hidden)))
CenteredRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray) = CenteredRBM(RBM(visible, hidden, w))
"""
    CenteredBinaryRBM(a, b, w, λv = 0, λh = 0)

Construct a centered binary RBM. The energy function is given by:

```math
E(v,h) = -a' * v - b' * h - (v - λv)' * w * (h - λh)
```
"""
function CenteredBinaryRBM(
        a::AbstractArray, b::AbstractArray, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray
    )
    return CenteredRBM(BinaryRBM(a, b, w), offset_v, offset_h)
end

function CenteredBinaryRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    return CenteredRBM(BinaryRBM(a, b, w))
end

"""
    uncenter(centered_rbm::CenteredRBM)

Constructs a plain `RBM` equivalent to `centered_rbm` (energies differ by the constant
given in [`center`](@ref), whose inverse this is). To construct an `RBM` that simply
neglects the offsets, call `RBM(centered_rbm)` instead.
"""
uncenter(centered_rbm::CenteredRBM) = unstandardize(centered_rbm)
uncenter(rbm::RBM) = rbm

@doc raw"""
    center(rbm::RBM, offset_v = 0, offset_h = 0)

Constructs a `CenteredRBM` equivalent to the given `rbm`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - E_c(v,h) = \sum_{i\mu}w_{i\mu}\lambda_i\lambda_\mu
```

where ``E(v,h)`` is the energy assigned by the original `rbm`, and
``E_c(v,h)`` is the energy assigned by the returned `CenteredRBM`.

This is the inverse operation of [`uncenter`](@ref).

To construct a `CenteredRBM` that simply includes these offsets,
call `CenteredRBM(rbm, offset_v, offset_h)` instead.
"""
center(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray) = center(center(rbm), offset_v, offset_h)
# centering is standardization that keeps the unit scales
center(rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray) =
    standardize(rbm, offset_v, offset_h, rbm.scale_v, rbm.scale_h)
center(rbm::CenteredRBM) = center(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h))
center(rbm::RBM) = CenteredRBM(rbm)

"""
    center!(centered_rbm, offset_v = 0, offset_h = 0)

Transforms the offsets of `centered_rbm`. The transformed model is equivalent to
the original one (energies differ by a constant).
"""
center!(rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray) =
    standardize!(rbm, offset_v, offset_h, rbm.scale_v, rbm.scale_h)
center!(rbm::CenteredRBM) = center!(rbm, Zeros(rbm.offset_v), Zeros(rbm.offset_h))

"""
    center_visible_from_data!(rbm::CenteredRBM, data; [wts])

Sets the visible offsets to the mean of `data`. The model is unchanged (energies
differ by a constant).
"""
function center_visible_from_data!(
        rbm::CenteredRBM, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data)
    )
    offset_v = batchmean(rbm.visible, data; wts)
    return standardize_visible!(rbm, offset_v, rbm.scale_v)
end

"""
    center_hidden_from_data!(rbm::CenteredRBM, data; [wts], damping = 1)

Sets the hidden offsets to the mean hidden activations conditioned on `data`.
The model is unchanged (energies differ by a constant).
"""
function center_hidden_from_data!(
        rbm::CenteredRBM, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), damping::Real = 1
    )
    offset_h_new = total_mean_h_from_v(rbm, data; wts)
    offset_h = (1 - damping) .* rbm.offset_h .+ damping .* offset_h_new
    return standardize_hidden!(rbm, offset_h, rbm.scale_h)
end

"""
    center_from_data!(rbm::CenteredRBM, data; [wts])

Sets the visible and hidden offsets from the means of `data`. The model is unchanged
(energies differ by a constant).
"""
function center_from_data!(
        rbm::CenteredRBM, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data)
    )
    center_visible_from_data!(rbm, data; wts)
    center_hidden_from_data!(rbm, data; wts)
    return rbm
end

"""
    rescale_hidden!(rbm::CenteredRBM, λ::AbstractArray)

Scales parameters such that hidden unit activations are divided by `λ`, preserving
the modeled distribution. This assumes the hidden units have a scale parameter,
otherwise it does nothing and returns `false`. Since the interaction involves
`h - offset_h`, the hidden offsets are divided by `λ` together with the activations, and
since the unit scales stay fixed, the weights absorb the factor `λ`.
"""
function rescale_hidden!(rbm::CenteredRBM, λ::AbstractArray)
    @assert size(rbm.hidden) == size(λ)
    if rescale_activations!(rbm.hidden, λ)
        rbm.w .*= _along_hidden(rbm, λ)
        rbm.offset_h ./= λ
        return true
    end
    return false
end

function initialize!(rbm::CenteredRBM, data::AbstractArray; ϵ::Real = 1.0e-6)
    initialize!(RBM(rbm), data; ϵ)
    center_from_data!(rbm, data)
    return rbm
end

"""
    pcd!(rbm::CenteredRBM, data; hidden_offset_damping = 1 // 100, kwargs...)

[`pcd!`](@ref) for a `CenteredRBM`, with the same keywords as for a plain `RBM`. The
visible offsets are set to the data means before training, and after every update the
hidden offsets move towards the minibatch conditional means by a fraction
`hidden_offset_damping`.
"""
function pcd!(
        rbm::CenteredRBM,
        data::AbstractArray;
        batchsize::Int = 1,
        iters::Int = 1,
        wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data),
        steps::Int = 1,
        optim::AbstractRule = Adam(),
        moments = moments_from_samples(rbm.visible, data; wts),
        hidden_offset_damping::Real = 1 // 100,
        l2_fields::Real = 0,
        l1_weights::Real = 0,
        l2_weights::Real = 0,
        l2l1_weights::Real = 0,
        zerosum::Bool = true,
        rescale::Bool = true,
        callback = Returns(nothing),
        vm::AbstractArray = _default_fantasy_chains(rbm, min(batchsize, size(data)[end])),
        shuffle::Bool = true,
        ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
        state = setup(optim, ps),
    )
    wts_mean, batchsize = _pcd_check_args(rbm, data, wts, batchsize)

    center_from_data!(rbm, data; wts) # initial centering from data
    # initial gauge; zerosum! first because rescaling preserves the zero-sum gauge,
    # while zerosum! perturbs weight norms
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        state, ps, ∂ = _pcd_step!(
            rbm, ps, state, vd, wd, vm, wts_mean;
            steps, moments, l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum
        )

        # damped update of the hidden offsets towards <h>_d from the minibatch
        center_hidden_from_data!(rbm, vd; wts = wd, damping = hidden_offset_damping)

        # reset gauge (zerosum! first, as above)
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)
    end
    return state, ps
end
