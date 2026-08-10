# default initialization of the persistent fantasy chains used by the PCD trainers
function _default_fantasy_chains(rbm, batchsize::Int)
    return sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize))
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
  `ArgumentError` — drop observations meant to be excluded (and their weights)
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
- `shuffle::Bool=true`: whether to reshuffle samples between epochs.
- `ps`: optimized parameter container. By default, this contains the visible,
  hidden, and interaction parameters.
- `state=setup(optim, ps)`: optimizer state.

Returns `(state, ps)`.
"""
function pcd!(
        rbm::RBM,
        data::AbstractArray;
        batchsize::Int = 1,
        iters::Int = 1, # number of gradient updates
        wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data), # data weights
        steps::Int = 1, # MC steps to update fantasy chains
        optim::AbstractRule = Adam(), # optimizer rule
        moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

        # regularization
        l2_fields::Real = 0, # visible fields L2 regularization
        l1_weights::Real = 0, # weights L1 regularization
        l2_weights::Real = 0, # weights L2 regularization
        l2l1_weights::Real = 0, # weights L2/L1 regularization

        # gauge
        zerosum::Bool = true, # zerosum gauge for Potts layers
        rescale::Bool = true, # normalize weights to unit norm (for continuous hidden units only)

        callback = Returns(nothing), # called for every batch

        # init fantasy chains
        vm::AbstractArray = _default_fantasy_chains(rbm, min(batchsize, size(data)[end])),

        shuffle::Bool = true,

        # parameters to optimize
        ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
        state = setup(optim, ps),
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    _validate_layer_parameters(rbm)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    size(data, ndims(data)) > 0 ||
        throw(ArgumentError("data must contain at least one sample"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    validate_wts(wts)
    wts_mean = mean(wts)
    batchsize = min(batchsize, length(wts))

    # initial gauge; zerosum! first because rescaling preserves the zero-sum gauge,
    # while zerosum! perturbs weight norms
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # positive phase
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)

        # negative phase: update persistent fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂m = ∂free_energy(rbm, vm)

        # weighted minibatch bias correction, in the gradient eltype
        batch_weight = convert(float(real(eltype(∂d.w))), mean(wd) / wts_mean)
        ∂ = (∂d - ∂m) * batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _validate_layer_parameters(rbm)

        # reset gauge (zerosum! first, as above)
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)
    end
    return state, ps
end
