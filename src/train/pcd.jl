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
  `callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)`. Slurp unused
  keywords with a trailing `_...`.
- `vm=nothing`: initial fantasy particles. By default, these are sampled after
  validating the model's pReLU parameters.
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
        vm = nothing,

        shuffle::Bool = true,

        # parameters to optimize
        ps = nothing,
        state = nothing,
    )
    _validate_layer_parameters(rbm)
    isnothing(vm) && (vm = _default_fantasy_chains(rbm, batchsize))
    reset_gauge! = () -> begin
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)
        return nothing
    end
    return _train!(
        rbm, data;
        batchsize, iters, wts, moments, optim, ps, state, shuffle,
        l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum, callback,
        setup! = (data, wts) -> reset_gauge!(),
        negative_phase = vd -> _pcd_negative_phase(rbm, vm, steps),
        post_update! = (vd, wd, ∂d) -> reset_gauge!(),
    )
end
