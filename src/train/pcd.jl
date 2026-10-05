# default initialization of the persistent fantasy chains used by the PCD trainers
function _default_fantasy_chains(rbm, batchsize::Int)
    return sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize))
end

# Argument checks shared by the `pcd!` trainers. Returns the mean data weight and the
# effective batch size.
function _pcd_check_args(rbm, data::AbstractArray, wts::AbstractVector, batchsize::Int)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    _validate_layer_parameters(rbm)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    size(data, ndims(data)) > 0 ||
        throw(ArgumentError("data must contain at least one sample"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    validate_wts(wts)
    return mean(wts), min(batchsize, length(wts))
end

"""
    pcd!(rbm, data; kwargs...)

Train an `RBM` with Persistent Contrastive Divergence (PCD).

`pcd!` repeatedly draws mini-batches from `data`, performs `steps` Gibbs updates
of persistent fantasy particles, estimates the positive/negative phase gradients,
applies optional regularization and gauge constraints, and updates model
parameters with an `Optimisers.jl` rule.

`data` must have shape `(size(rbm.visible)..., nsamples)`.

# Keyword arguments
- `batchsize::Int=1`: number of samples per update.
- `iters::Int=1`: number of parameter updates.
- `wts::AbstractVector{<:Real}`: finite, positive per-sample
  weights, lazy uniform weights by default. Zero or negative weights raise an
  `AssertionError` — drop observations meant to be excluded (and their weights)
  beforehand. Callbacks receive the minibatch weights as `wd`.
- `steps::Int=1`: Gibbs steps used to update persistent chains each iteration.
- `optim::AbstractRule=Adam()`: optimizer rule from `Optimisers.jl`.
- `moments=moments_from_samples(rbm.visible, data; wts)`: data moments used
  by the positive phase.
- `l2_fields::Real=0`: L2 regularization on visible fields.
- `l1_weights::Real=0`: L1 regularization on interaction weights.
- `l2_weights::Real=0`: L2 regularization on interaction weights.
- `l2l1_weights::Real=0`: group-like L2/L1 weight regularization.
- `zerosum::Bool=true`: enforce zero-sum gauge on Potts layers.
- `rescale::Bool=true`: rescale weights (mainly useful for continuous hidden units).
- `callback=Returns(nothing)`: called after every update as
  `callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)`. Slurp unused
  keywords with a trailing `_...`.
- `vm`: initial fantasy particles. By default, `min(batchsize, nsamples)`
  chains sampled from the visible layer with zero inputs.
- `ps`: optimized parameter container. By default, this contains the visible,
  hidden, and interaction parameters.
- `state=setup(optim, ps)`: optimizer state.

Returns `(state, ps)`.

A plain `RBM` is trained as the equivalent `StandardizedRBM` whose offsets and scales are
fixed to zero and one (see the [`pcd!`](@ref) method for `StandardizedRBM`), so it also
accepts the standardization keywords `damping`, `ϵv`, `ϵh` and
`regularize_unstandardized`, which have no effect on it.
"""
function pcd!(rbm::RBM, data::AbstractArray; callback = Returns(nothing), kwargs...)
    std_rbm = PlainStandardizedRBM(rbm) # shares the layers and weights of `rbm`
    # the callback receives `rbm` rather than `std_rbm` (the rightmost keyword wins)
    return pcd!(std_rbm, data; callback = (; kw...) -> callback(; kw..., rbm), kwargs...)
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

        # positive phase
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)

        # negative phase: update persistent fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂m = ∂free_energy(rbm, vm)

        # weighted minibatch bias correction, in the gradient eltype
        batch_weight = convert(float(real(eltype(∂d.w))), mean(wd) / wts_mean)
        ∂ = (∂d - ∂m) * batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum, regularize_unstandardized)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _validate_layer_parameters(rbm)

        # these leave the distribution unchanged
        standardize_hidden_from_v!(rbm, vd; wts = wd, damping, ϵ = ϵh)
        zerosum && zerosum!(rbm)
        rescale && rescale_hidden_activations!(rbm)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)
    end
    return state, ps
end
