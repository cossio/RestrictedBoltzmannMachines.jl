```@meta
CurrentModule = RestrictedBoltzmannMachines
```

# [Training RBMs with `pcd!`](@id training)

This page describes how model training works in this package, focusing on:

- [`pcd!`](@ref) for plain `RBM`,
- specialized [`pcd!`](@ref) for `CenteredRBM` and `StandardizedRBM` (stdRBM).

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
weights through `wts`. Weights must be finite, real,
and nonnegative, and at least one weight must be positive.

Observations with zero weight are excluded before data moments and mini-batches
are formed. They therefore do not advance persistent or coupled chains, reach
the optimizer or callback, or contribute to centered/standardized wrapper
statistics. This is equivalent to removing those observations and their
weights before training.

Training calculations rescale positive weights internally without changing
their relative values, avoiding overflow for finite extreme weights. Callbacks
still receive the original positive mini-batch weights in `wd`.

The `iters` argument always counts completed parameter updates. Ignored
zero-weight observations do not consume iterations, and callbacks receive
consecutive `iter` values from `1` through `iters`.

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
  - `wts`: optional finite, real, nonnegative sample weights,
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

where `wd` are the weights of the current mini-batch (`nothing` if `wts` was not
given); zero weights do not appear in these mini-batches. Define callbacks with
a trailing `_...` slurp (e.g.
`callback(; rbm, iter, _...) = ...`) to stay robust if more keywords are added.

## Practical tuning guidelines

- Start with `steps=1`; increase only if fantasy chains mix too slowly.
- Use `batchsize` large enough for stable gradients but small enough for memory limits.
- Track a metric in `callback` every N iterations instead of every update if evaluation is expensive.
- When using stdRBM, tune `damping` conservatively (small values adapt statistics more smoothly).
