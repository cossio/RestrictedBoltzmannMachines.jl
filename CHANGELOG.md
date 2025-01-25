# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 3.7.0

- Move functionality from `StandardizedRestrictedBoltzmannMachines.jl` to this package, through the Extensions mechanism.

## 3.6.0

- Move functionality from `RestrictedBoltzmannMachinesHDF5.jl` to this package, through the Extensions mechanism.

## 3.5.0

- Move functionality from `CudaRBMs.jl` to this package, through the Extensions mechanism.
- Move `PottsGumbel` to this package.

## 3.4.1

- Use `default_rng` in place of `GLOBAL_RNG`, which fixes an issue with CUDA [e84438a](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/e84438a9d0ef04ad981cd552184c79a1d90dd45f).

## 3.2.5

- Compat with Optimisers v0.3.

## 3.2.0

- Patch CUDA issue with Potts due to https://github.com/JuliaGPU/CUDA.jl/issues/1957.
- `callback` now receives `state` too.

## 3.1.0

- `visible_cgf`, `free_energy_h`, `free_energy_v`, and gradients.

## 3.0.0

- BREAKING: `rescale_hidden!` returns `true` or `false` depending on whether the hidden units were rescaled or not.

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
