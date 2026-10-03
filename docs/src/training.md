```@meta
CurrentModule = RestrictedBoltzmannMachines
```

# [Training RBMs with `pcd!`](@id training)

This page describes how model training works in this package, focusing on:

- [`pcd!`](@ref) for plain `RBM`,
- specialized [`pcd!`](@ref) for `CenteredRBM` and `StandardizedRBM` (stdRBM),
- [`ptt!`](@ref), equilibrium training by Parallel Trajectory Tempering.

Unbiased Contrastive Divergence (`ucd!`) for binary-binary RBMs lives in a
separate package, [ucdRBMs.jl](https://github.com/cossio/ucdRBMs.jl).

See also the [MNIST example](@ref MNIST) for an end-to-end runnable script.

## Training workflow

The usual training workflow is:

1. Build an RBM (`BinaryRBM`, `GaussianRBM`, `PottsRBM`, ...).
2. Prepare data with shape `(size(rbm.visible)..., nsamples)`.
3. Call [`initialize!`](@ref) (for plain RBMs) or `standardize(...)` if using stdRBM.
4. Train with [`pcd!`](@ref).
5. Monitor training with [`log_pseudolikelihood`](@ref), [`reconstruction_error`](@ref), or a callback.

## Weighted data

Plain, centered, and standardized [`pcd!`](@ref) accept optional per-sample
weights through `wts`. Weights must be finite, positive reals: zero or
negative weights raise an `ArgumentError`. Observations meant to be excluded
must be dropped (with their weights) before calling [`pcd!`](@ref).

The `iters` argument always counts completed parameter updates, and callbacks
receive consecutive `iter` values from `1` through `iters`.

## How `pcd!` works (plain `RBM`)

At each training iteration, [`pcd!`](@ref) on `RBM`:

1. draws a mini-batch from data,
2. advances persistent fantasy particles by Gibbs updates (`steps`),
3. computes positive and negative phase gradients (`∂d - ∂m`),
4. applies optional regularization,
5. updates parameters through an `Optimisers.jl` rule,
6. reapplies gauge constraints (`zerosum!`, `rescale_weights!` when enabled).

### Important arguments for `pcd!(rbm::RBM, data; ...)`

- Optimization:
  - `iters`: number of completed parameter updates,
  - `batchsize`: mini-batch size,
  - `optim`: optimizer rule (default `Adam()`),
  - `state`, `ps`: optimizer internals/state containers.
- Sampling:
  - `steps`: Gibbs steps for fantasy-chain updates per iteration,
  - `vm`: initial fantasy particles.
- Data handling:
  - `shuffle`: reshuffle data between epochs,
  - `wts`: optional finite, positive sample weights,
  - `moments`: data sufficient statistics (defaults to layer moments from `data`).
- Regularization:
  - `l2_fields`, `l1_weights`, `l2_weights`, `l2l1_weights`.
- Gauge:
  - `zerosum` (Potts-family gauge),
  - `rescale` (weight normalization, mainly relevant for continuous hidden units).
- Monitoring:
  - `callback`: called at every update as `callback(; rbm, optim, state, iter, vm, vd, wd)`.

## Specialized `pcd!` for `StandardizedRBM` (stdRBM)

`pcd!(rbm::StandardizedRBM, data; ...)` follows the same PCD backbone, with extra
steps to keep the standardized parameterization stable during learning.

In addition to the standard PCD updates, it:

1. updates visible standardization from data (`standardize_visible_from_data!`),
2. updates hidden standardization from current mini-batches (`standardize_hidden_from_v!`),
3. optionally rescales hidden activations (`rescale_hidden_activations!`),
4. can regularize either standardized or unstandardized parameters.

### stdRBM-specific arguments

- Standardization controls:
  - `damping`: smoothing factor for hidden standardization updates (`0 ≤ damping ≤ 1`),
  - `ϵv`, `ϵh`: pseudocount-like stabilizers for visible/hidden standardization.
- Standardization-aware regularization:
  - `regularize_unstandardized`: if `true`, regularization is applied in the unstandardized gauge.
- Hidden rescaling:
  - `rescale_hidden`: absorb scale into hidden activation when relevant.

Other common arguments remain the same (`iters`, `batchsize`, `steps`, `optim`,
`wts`, `l1_weights`, `l2_weights`, `l2_fields`, `l2l1_weights`, `zerosum`,
`callback`, `vm`).

The stdRBM callback is called as:

`callback(; rbm, optim, state, ps, iter, vm, vd, wd, ∂)`

where `wd` are the weights of the current mini-batch (lazy uniform `Ones` if
`wts` was not given). Define callbacks with a trailing `_...` slurp (e.g.
`callback(; rbm, iter, _...) = ...`) to stay robust if more keywords are added.

## Equilibrium training with `ptt!`

On multimodal or scarce data, the persistent chains of [`pcd!`](@ref) can fall out of
equilibrium (for instance, getting trapped in some of the modes), which biases the
gradient. [`ptt!`](@ref) implements Parallel Trajectory Tempering
([Béreux, Decelle, Furtlehner, Seoane, 2026](https://arxiv.org/abs/2607.27077)), which
keeps the chains at equilibrium by replica exchange with frozen checkpoints of the
training trajectory itself:

```julia
rbm = BinaryRBM(Float32, 784, 200)
initialize!(rbm, data)
ladder = TrajectoryLadder(rbm; nchains = 1000)
ptt!(rbm, data; ladder, batchsize = 500, iters = 10_000, steps = 10)
log_likelihood(ladder, data) # with the partition function estimated by the ladder
```

The [`TrajectoryLadder`](@ref) holds the checkpoints and the persistent chains:

1. It starts from the independent-site model obtained by setting the weights to zero,
   which is sampled exactly and has a known partition function, and builds checkpoints
   along the weights scaled down from those of `rbm`, ending with `rbm` itself.
2. Each training update proposes to exchange every chain with an equilibrium sample of
   the last checkpoint, drawn from a reservoir, and then runs `steps` Gibbs steps.
3. When the swap acceptance between the last checkpoint and the model falls below `α`
   (default `0.3`), the model is frozen as a new checkpoint. Its chains are
   thermalized by exchanges with the reservoir of the previous checkpoint (for 20
   autocorrelation times of the exchanges) and then collected into a new reservoir.
4. Log-partition functions of successive checkpoints are linked by the Bennett
   acceptance ratio, which gives [`log_partition(ladder)`](@ref log_partition(::TrajectoryLadder))
   and [`log_likelihood(ladder, v)`](@ref log_likelihood(::TrajectoryLadder, ::AbstractArray))
   at no extra cost.
5. If an update makes the acceptance drop below `αmin` (default `0.1`), or below `α`
   right after a checkpoint, it is rejected: the model is restored to the last
   checkpoint and the learning rate of the optimizer is halved.

`ptt!` trains plain `RBM`s and accepts the keywords of [`pcd!`](@ref) for `RBM`, plus the
`ladder`; its callback also receives the ladder as `ladder`. A `TrajectoryLadder` can also
be built for a `CenteredRBM` or `StandardizedRBM`, to sample it or estimate its partition
function. The paper uses [`CossimDescent`](@ref), a gradient descent whose learning rate
adapts to the alignment of successive gradients; the default optimiser is `Adam()`, as for
`pcd!`. The
size of the updates sets how often checkpoints are frozen, each of which costs some tens
of sweeps, so a smaller learning rate trades slower learning for fewer checkpoints.
Every checkpoint is kept in `ladder.checkpoints`, with its log-partition function in
`ladder.logZ`. The log-partition functions accumulate the errors of the successive
estimates, from a few hundredths to about a tenth of a nat per checkpoint (slightly biased
downwards, since equilibrium samples collected from the persistent chains are correlated),
so fewer checkpoints give more accurate estimates. For a final, independent estimate,
build a new `TrajectoryLadder` for a copy of the trained model.

## Practical tuning guidelines

- Start with `steps=1`; increase only if fantasy chains mix too slowly.
- Use `batchsize` large enough for stable gradients but small enough for memory limits.
- Track a metric in `callback` every N iterations instead of every update if evaluation is expensive.
- When using stdRBM, tune `damping` conservatively (small values adapt statistics more smoothly).
