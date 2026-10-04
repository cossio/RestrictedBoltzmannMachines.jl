```@meta
CurrentModule = RestrictedBoltzmannMachines
```

# [Training RBMs with `pcd!`](@id training)

This page describes how model training works in this package, focusing on:

- [`pcd!`](@ref) for plain `RBM`,
- [`pcd!`](@ref) for `StandardizedRBM` (stdRBM), including `CenteredRBM`,
- [`ptt!`](@ref), equilibrium training by Parallel Trajectory Tempering.

There is one trainer: a plain `RBM` is trained as the equivalent stdRBM whose offsets and
scales are fixed to zero and one, so the extra stdRBM steps below reduce to nothing for it.

Unbiased Contrastive Divergence (`ucd!`) for binary-binary RBMs lives in a
separate package, [ucdRBMs.jl](https://github.com/cossio/ucdRBMs.jl).

See also the [MNIST example](@ref MNIST) for an end-to-end runnable script.

## Training workflow

The usual training workflow is:

1. Build an RBM (`BinaryRBM`, `GaussianRBM`, `HopfieldRBM`, or `RBM(visible, hidden, w)` for any pair of layers).
2. Prepare data with shape `(size(rbm.visible)..., nsamples)`.
3. Call [`initialize!`](@ref) once on the data. For a stdRBM this also sets the offsets and
   scales from the data (`standardize(...)` to choose them instead).
4. Train with [`pcd!`](@ref).
5. Monitor training with [`log_pseudolikelihood`](@ref), [`reconstruction_error`](@ref), or a callback.

## Weighted data

Plain, centered, and standardized [`pcd!`](@ref) accept optional per-sample
weights through `wts`. Weights must be finite, positive reals: zero or
negative weights fail validation with an `AssertionError`. Drop observations
meant to be excluded (with their weights) before calling [`pcd!`](@ref).

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
  - `iters`: number of completed parameter updates (callbacks see `iter = 1:iters`),
  - `batchsize`: mini-batch size,
  - `optim`: optimizer rule (default `Adam()`),
  - `state`, `ps`: optimizer internals/state containers.
- Sampling:
  - `steps`: Gibbs steps for fantasy-chain updates per iteration,
  - `vm`: initial fantasy particles.
- Data handling:
  - `wts`: optional finite, positive sample weights,
  - `moments`: data sufficient statistics (defaults to layer moments from `data`).
- Regularization:
  - `l2_fields`, `l1_weights`, `l2_weights`, `l2l1_weights`.
- Gauge:
  - `zerosum` (Potts-family gauge),
  - `rescale` (weight normalization, mainly relevant for continuous hidden units).
- Monitoring:
  - `callback`: called at every update as `callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)`,
    where `wd` are the weights of the current mini-batch (lazy uniform weights if `wts`
    was not given) and `∂` the gradient. Define callbacks with a trailing `_...` slurp
    (e.g. `callback(; rbm, iter, _...) = ...`) to stay robust if more keywords are added.

## Specialized `pcd!` for `StandardizedRBM` (stdRBM)

`pcd!(rbm::StandardizedRBM, data; ...)` follows the same PCD backbone, with extra
steps to keep the standardized parameterization stable during learning. It also trains a
`CenteredRBM`, the special case whose scales stay fixed to one, so that only its offsets
are fitted.

In addition to the standard PCD updates, it:

1. sets the visible and hidden standardization from data before training
   (`standardize_visible_from_data!`, `standardize_hidden_from_v!`),
2. updates the hidden standardization from current mini-batches (`standardize_hidden_from_v!`),
3. optionally fixes the scale gauge of the hidden units (`rescale_hidden_activations!`:
   absorbing `scale_h` for a `StandardizedRBM`, normalizing the weights for a `CenteredRBM`),
4. can regularize either standardized or unstandardized parameters.

### stdRBM-specific arguments

- Standardization controls:
  - `damping`: smoothing factor for hidden standardization updates (`0 ≤ damping ≤ 1`),
  - `ϵv`, `ϵh`: pseudocount-like stabilizers for visible/hidden standardization.
- Standardization-aware regularization:
  - `regularize_unstandardized`: if `true`, regularization is applied in the unstandardized gauge.

Other arguments, including `rescale` and `callback`, are the same as for a plain `RBM`.
A plain `RBM` also accepts the stdRBM-specific arguments, which have no effect on it.

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
   thermalized by exchanges with the reservoir of the previous checkpoint, for 20
   autocorrelation times of the ladder level of the chains, as in the paper, and then
   collected into a new reservoir.
4. Log-partition functions of successive checkpoints are linked by the Bennett
   acceptance ratio, which gives [`log_partition(ladder)`](@ref log_partition(::TrajectoryLadder))
   and [`log_likelihood(ladder, v)`](@ref log_likelihood(::TrajectoryLadder, ::AbstractArray))
   at no extra cost.
5. An update is rejected if it makes the acceptance drop below `αmin` (default `0.1`), or
   below `α` right after a checkpoint, or if the chains of a new checkpoint do not reach
   an acceptance of `αmin` with the previous one once thermalized. The model is then
   restored to the last checkpoint, the learning rate of the optimiser is halved, and the
   optimiser forgets its momenta. As in the reference implementation, the halving is
   permanent, and `ptt!` warns once it has halved the learning rate 10 times. On
   clustered data, an optimiser with momentum, such as `Nesterov` or `Adam`, can make
   the model oscillate across a phase transition, so that its updates keep being
   rejected, whatever its initial learning rate; plain gradient descent (`Descent`)
   can then train through.

`ptt!` trains plain `RBM`s from scratch, starting from [`initialize!`](@ref), and
accepts the keywords of [`pcd!`](@ref) for `RBM` except `shuffle`, plus the `ladder`; its
callback also receives the ladder as `ladder`. Unlike `pcd!`, and as in the reference
implementation, it draws every minibatch at random, independently of the others: the
minibatches of an epoch have anticorrelated noise, which makes the learning rate of
`CossimDescent` collapse when epochs have few minibatches. The optimiser must have a
learning rate `eta`. To continue training, pass the ladder of the previous run: without
it, `ptt!` builds a new one. A `TrajectoryLadder` can also be built for a `CenteredRBM` or
`StandardizedRBM`, to sample it or estimate its partition function. The paper uses
[`CossimDescent`](@ref), a gradient descent whose learning rate adapts to the alignment
of successive gradients. The default optimiser is `Adam(1e-4)`, a tenth of the learning
rate of `pcd!`'s `Adam()`, whose larger steps can be rejected from the first update. On
four protein and RNA families, it reached the best or tied best validation
log-likelihood after 100k updates, compared with `Descent(1e-2)` and `CossimDescent`, at
the cost of more checkpoints.

The size of the updates sets how often checkpoints are frozen, each of which costs some
tens of sweeps, so a smaller learning rate trades slower learning for fewer checkpoints.
Repeated rejections at the same checkpoint can mean that the chains lag behind the model,
so that the acceptance overestimates its overlap with the checkpoint. A larger `steps`
helps if that many Gibbs steps move the chains between the modes of the model. Every
checkpoint is kept in `ladder.checkpoints`, with its log-partition function in
`ladder.logZ`. The log-partition functions accumulate the errors of the successive
estimates, from a few hundredths to about a tenth of a nat per checkpoint, so fewer
checkpoints give more accurate estimates.

## Practical tuning guidelines

- Start with `steps=1`; increase only if fantasy chains mix too slowly.
- Use `batchsize` large enough for stable gradients but small enough for memory limits.
- Track a metric in `callback` every N iterations instead of every update if evaluation is expensive.
- When using stdRBM, tune `damping` conservatively (small values adapt statistics more smoothly).
