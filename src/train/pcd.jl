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
- `wts::Union{AbstractVector,Nothing}=nothing`: optional per-sample weights.
- `steps::Int=1`: Gibbs steps used to update persistent chains each iteration.
- `optim::AbstractRule=Adam()`: optimizer rule from `Optimisers.jl`.
- `moments=moments_from_samples(rbm.visible, data; wts)`: data moments used by
  the positive phase.
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
- `ps=(; visible=rbm.visible.par, hidden=rbm.hidden.par, w=rbm.w)`: optimized
  parameter container.
- `state=setup(optim, ps)`: optimizer state.

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
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps) # initialize optimiser state
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)
    _check_prelu_eta(rbm.visible, rbm.hidden, :pcd_start)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    # store average weight of each data point
    wts_mean = isnothing(wts) ? 1 : mean(wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # correct weighted minibatch bias
        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
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
