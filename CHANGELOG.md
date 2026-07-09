# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- `nsReLU` is now implemented as a special case of `xReLU`. `xReLU` gained a third `Bool` type parameter (`xReLU{N,A,FixGamma}`) that, when `true`, fixes the scale γ = 1 and omits it from `par` (which then stores only θ, Δ, ξ). `nsReLU{N,A}` is now an alias for `xReLU{N,A,true}`, so `layer isa xReLU` is `true` for `nsReLU` layers, and `layer.γ` on an `nsReLU` returns a lazy array of ones. Public constructors, the `par` layout, gradient shapes, and the HDF5 format are unchanged. Constructing the trainable-γ variant directly as `xReLU{N,A}(par)` no longer works (use `xReLU(par)` or `xReLU{N,A,false}(par)`).

## 5.4.0

- Added `ucd!` trainer (Unbiased Contrastive Divergence) and `unbiased_sample` sampler.
- Restored support for Julia LTS (v1.10).
- `log_pseudolikelihood` is now GPU friendly (no scalar indexing) [#101](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/101).
- Layers, RBMs (including `CenteredRBM`, `StandardizedRBM`) and `∂RBM` now implement the [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) interface, so generic device transfers like `adapt(CuArray, rbm)` work with any GPU array backend. `gpu`/`cpu` are now generic methods built on Adapt, and `CenteredRBM` gains GPU transfer. Added GPU-semantics tests that run on CI without GPU hardware, using JLArrays with `allowscalar(false)` [#102](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/102).

## 5.3.2

- Fixed weighted centered PCD initialization and empty `shuffleobs` inputs [a33d7b4](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a33d7b441507b743c55f7af57d681bcca2756418).

## 5.3.1

- Speed up `log_pseudolikelihood` by reusing hidden inputs and local free-energy updates [d01bacb](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/d01bacbec488d32fece46d15655ea97933265b14).

## 5.3.0

- Added HDF5 support for `nsReLU` layer.

## 5.2.0

- Added AIS support for `nsReLU` layer.

## 5.1.1

- Internal formatting and import cleanup; no functional changes.

## 5.1.0

- Added `nsReLU` layer.

## 5.0.0

- BREAKING: removed `hidden_statistics_from_inputs`. This was a duplicate of `total_meanvar_from_inputs`.
- Added `rescale_weights!` for `StandardizedRBM`. This normalizes the unstandardized weights attached to each hidden unit.

## 4.2.1

- Fixed `regularization_penalty_fields` for `dReLU` layers, which now correctly sums `abs2(θp)` and `abs2(θn)` instead of only `abs2(θ)`.
- Added `regularization_penalty` method for `StandardizedRBM`, computing the penalty on the equivalent unstandardized parameters.

## 4.2.0

- Added `regularization_penalty`.

## 4.1.0

- Added methods: `total_mean_h_from_v`, `total_mean_v_from_h`, `total_var_h_from_v`, `total_var_v_from_h`.

## 4.0.0

- BREAKING: Changed the way regularization works for standardized and centered RBMs. Now regularization acts as if it was applied to the equivalent normal RBM parameters, whereas before it was applied directly to the weights of the standardized/centered parameterizations.

## 3.9.0

- Support Julia LTS (v1.10).
- Add `∂free_energy_h` method for standardized RBMs.

## 3.8.1

- Fixed CUDA `gpu`/`cpu` dispatch for `StandardizedRBM`, which previously referenced the wrong module prefix.

## 3.8.0

- Move functionality from `CenteredRBMs.jl` to this package.

## 3.7.1

- Added CUDA `gpu`/`cpu` support for `StandardizedRBM`.

## 3.7.0

- Move functionality from `StandardizedRestrictedBoltzmannMachines.jl` to this package.

## 3.6.0

- Move functionality from `RestrictedBoltzmannMachinesHDF5.jl` to this package, through the Extensions mechanism.

## 3.5.0

- Move functionality from `CudaRBMs.jl` to this package, through the Extensions mechanism.
- Move `PottsGumbel` to this package.

## 3.4.1

- Use `default_rng` in place of `GLOBAL_RNG`, which fixes an issue with CUDA [e84438a](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/e84438a9d0ef04ad981cd552184c79a1d90dd45f).

## 3.4.0

- Added `SpinRBM` convenience constructor.

## 3.2.5

- Compat with Optimisers v0.3.

## 3.2.0

- Patch CUDA issue with Potts due to https://github.com/JuliaGPU/CUDA.jl/issues/1957.
- `callback` now receives `state` too.

## 3.1.0

- `visible_cgf`, `free_energy_h`, `free_energy_v`, and gradients.

## 3.0.0

- BREAKING: `rescale_hidden!` returns `true` or `false` depending on whether the hidden units were rescaled or not.

## 2.3.0

- Compat with FillArrays v1.

## 2.2.0

- Initialization from data for dReLU, pReLU, xReLU.

## 2.1.1

- Fix mirror.

## 2.1.0

- Allow division of ∂RBM by a scalar.

## 2.0.4

- Always convert to weights eltype in ∂interaction_energy (previously, converted only in some cases).

## 2.0.3

- Convert to weights eltype in ∂interaction_energy (fix #10).

## 2.0.2

- Fix batch weight correction (https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/355b5faae78d268f083787c7f92a0e995eee6116).

## 2.0.1

- Close CUDA issue [#20](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/20).

## 2.0.0

- This CHANGELOG file.
- Rescale weights to norm 1, instead of hidden unit activities to unit variances. This is a simpler way to settle the scale degeneracy between weights and hidden unit activities for continuous hidden units. Commit [here](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/4cae554013d7b6ab97a900910ff67d2a43d263b0).
    * Introduce `rescale_weights!(rbm)` to normalize weights attached to each hidden units.
    * Now `pcd!(...; rescale=true, ...)` uses `rescale_weights!`, instead of scaling hidden unit activities to unit variances.
    * BREAKING: Removed `ρh`, `ϵh` keyword arguments from `pcd!`, which used to control the tracking of hidden unit variances during training.
    * BREAKING: `grad2var` has been removed.
- Allow passing `ps`, `state` to `pcd!` to control which parameters are optimized. Now `pcd!` returns `state, ps`, which can be breaking. Commit [here](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/05fade7e567f557dba457c287ca4ebf0faab14d4).

## 1.0.0

- Release v1.0.0.

## 0.39

- Now `pcd!(...; iters=n, ...)` performs `n` gradient updates. This replaces the `epochs` setting.
- `pcd` now uses the Optimisers framework instead of Flux. In particular, `optim` expects a `Optimisers.AbstractRule`.

## 0.38
