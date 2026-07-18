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
- `wts::Union{AbstractVector,Nothing}=nothing`: optional finite, nonnegative
  per-sample weights. Zero-weight samples are ignored, and at least one weight
  must be positive.
- `steps::Int=1`: Gibbs steps used to update persistent chains each iteration.
- `optim::AbstractRule=Adam()`: optimizer rule from `Optimisers.jl`.
- `moments=moments_from_samples(rbm.visible, data; wts)`: data moments used by
  the positive phase (zero-weight samples contribute nothing).
- `l2_fields::Real=0`: L2 regularization on visible fields.
- `l1_weights::Real=0`: L1 regularization on interaction weights.
- `l2_weights::Real=0`: L2 regularization on interaction weights.
- `l2l1_weights::Real=0`: group-like L2/L1 weight regularization.
- `zerosum::Bool=true`: enforce zero-sum gauge on Potts layers.
- `rescale::Bool=true`: rescale weights (mainly useful for continuous hidden units).
- `callback=Returns(nothing)`: called after every update as
  `callback(; rbm, optim, state, iter, vm, vd, wd)`.
- `vm=sample_from_inputs(...)`: initial fantasy particles.
- `shuffle::Bool=true`: whether to reshuffle samples between epochs.
- `ps=nothing`: optimized parameter container. By default, this contains the
  visible, hidden, and interaction parameters.
- `state=nothing`: optimizer state. By default, this is initialized after
  validating the model's pReLU parameters.

Returns `(state, ps)`.
"""
function pcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1, # number of gradient updates
    wts::Union{AbstractVector, Nothing} = nothing, # data weights
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
    vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),

    shuffle::Bool = true,

    # parameters to optimize
    ps = nothing,
    state = nothing,
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)
    _check_prelu_eta(rbm.visible, rbm.hidden, :pcd_start)
    isnothing(ps) &&
        (ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w))
    isnothing(state) && (state = setup(optim, ps))

    data, wts, normalization, batchsize = _prepare_training_data(data, wts; batchsize)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        batch_weight = _batch_weight(wd, normalization)

        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # correct weighted minibatch bias
        ∂ *= batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _check_prelu_eta(rbm.visible, rbm.hidden, :pcd_update)

        # reset gauge
        rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, state, iter, vm, vd, wd)
    end
    return state, ps
end
