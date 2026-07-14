# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RestrictedBoltzmannMachines.jl is a Julia package for training and inference with Restricted Boltzmann Machines (RBMs). It supports multiple layer types (Binary, Spin, Potts, Gaussian, ReLU variants), GPU acceleration via CUDA.jl, and HDF5 persistence. Requires Julia 1.12+.

## Common Commands

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file (uses test/ project for test-only deps like Zygote, QuadGK)
julia --project=test test/pcd.jl

# Start a REPL with the package loaded
julia --project=.
julia> using RestrictedBoltzmannMachines
```

## Architecture

### Array Dimension Convention

Data arrays have layer dimensions first and batch dimension last. For a layer of size `(N,)`, a single sample is a vector of length `N`, and a batch is a matrix `(N, B)`. For Potts layers of size `(Q, N)`, a batch is `(Q, N, B)`. The weight matrix `w` has shape `(size(visible)..., size(hidden)...)`. Functions like `energy`, `free_energy`, `sample_from_inputs` all follow this convention and broadcast over the trailing batch dimension.

### Type Hierarchy

`AbstractLayer{N}` (N = ndims) is the base type for all unit types. Each layer implements: `energy`, `cgfs`, `sample_from_inputs`, `mean_from_inputs`, `var_from_inputs`, `mode_from_inputs`.

All layers store parameters in a single `.par` array. The first dimension is the number of parameters of the layer type (e.g. 1 for Binary/Spin/Potts which have only `θ`, 2 for Gaussian which has `θ` and `γ`, 4 for dReLU). The remaining dimensions are the layer's spatial dimensions (the grid of units). So `ndims(par) == N + 1` where `N` is the layer ndims. Named parameter accessors (e.g. `layer.θ`, `layer.γ`) are views into `.par`.

Potts is special: it has `AbstractLayer{2}` because its first layer dimension is the one-hot (categorical) dimension with `Q` classes, and the second is the number of units. So `size(potts_layer) == (Q, N)` and `par` has shape `(1, Q, N)`.

Layer types: `Binary`, `Spin`, `Potts`, `Gaussian`, `ReLU`, `dReLU`, `pReLU`, `xReLU`, `nsReLU`, `PottsGumbel`.

`RBM{V,H,W}` holds `visible` layer, `hidden` layer, and weight matrix `w`. Extended by `CenteredRBM` (with offset parameters) and `StandardizedRBM` (with offset + scale parameters).

### Module Organization

- **`src/layers/`** — Layer type definitions. `abstractlayer.jl` defines the interface; each file implements one layer type; `common.jl` has shared utilities.
- **`src/rbms/`** — Convenience constructors (`BinaryRBM`, `HopfieldRBM`, etc.).
- **`src/train/`** — Training: `pcd.jl` (Persistent Contrastive Divergence), `initialization.jl` (data-driven init), `gradient.jl`.
- **`src/gauge/`** — Gauge transformations: `zerosum.jl`, `rescale_hidden.jl`, `shift_fields.jl`.
- **`src/util/`** — Utilities: linear algebra helpers, one-hot encoding, truncated normal sampling.
- **`ext/`** — Package extensions for CUDA (GPU) and HDF5 (save/load).

### Key Functions

- **Training:** `pcd!(rbm, data; ...)`, `initialize!(rbm, data)`
- **Sampling:** `sample_v_from_h`, `sample_h_from_v`, `sample_v_from_v`, `metropolis`
- **Evaluation:** `free_energy`, `energy`, `log_pseudolikelihood`, `log_partition`, `reconstruction_error`
- **Partition function:** `aise` (Annealed Importance Sampling), `raise` (reverse AIS)

### Workspace

The project uses Julia workspaces (`[workspace]` in Project.toml) with sub-projects: test, docs, notebooks, repl.

## Testing

Tests in `test/runtests.jl` are organized as independent modules (each wrapped in its own `module`). Each test file can be run standalone with `--project=test`. Tests use property-based testing across dimensions and gradient checking with Zygote and FiniteDifferences.

GPU compatibility is tested without GPU hardware in `test/jlarrays.jl`, using JLArrays with `allowscalar(false)`. Do not commit tests that require a physical GPU (GitHub CI has none); run those locally only.

## Releasing a new version

During development the version in Project.toml carries a `-DEV` suffix (e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in CHANGELOG.md. To release:

1. Make a single commit titled `vX.Y.Z` that drops the `-DEV` suffix from `version` in Project.toml and renames the `## Unreleased` CHANGELOG section to `## X.Y.Z`. Land it on `master` (directly, or via a PR like [#123](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/123)).
2. Create a frozen branch `release-X.Y.Z` pointing at the master commit that contains the release (the merge commit if it landed via PR) and push it: `git push origin <sha>:refs/heads/release-X.Y.Z`. Never commit to this branch — its whole point is that its HEAD *is* the release commit, so registration targets the exact commit even if master keeps moving in the meantime.
3. Comment on issue [#124 "Julia registration"](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124) — the single permanent issue used for all registrations, kept closed on purpose so it doesn't pollute the issue list (comments on closed issues still trigger Registrator; do NOT open a new issue per release). Include release notes (markdown, typically the CHANGELOG entries for this version) in the same comment:

   ```
   @JuliaRegistrator register branch=release-X.Y.Z

   Release notes:

   ## Breaking changes

   - blah
   ```

   Caveats learned the hard way: Registrator ignores PR comments entirely ("disabled", it replies); and from an issue comment only `branch=<name>` is accepted — there is no way to pin a SHA, which is why step 2 exists. Registrator resolves the branch to a concrete SHA when it processes the comment and replies with a link to the PR it opens in the General registry. The notes are added to that registry PR and to the GitHub release. Once the registry PR merges, TagBot creates the `vX.Y.Z` tag at the registered SHA and the GitHub release automatically. (If the register comment was posted without notes, re-invoke Registrator with them to update the registration — see [Registrator's reply](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a4dcb8cee859c752881c6c1bb6051edaffcecf84#commitcomment-188488609).)
4. Monitor the registration PR in the General registry until it merges. Registrator's reply to the triggering comment links it (e.g. [this reply](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124#issuecomment-4966310598) → [JuliaRegistries/General#161274](https://github.com/JuliaRegistries/General/pull/161274) for v5.7.0). AutoMerge normally merges it within ~15–30 minutes. Watch for AutoMerge failures (version-increment, compat or project-file checks) and for comments from registry maintainers requesting changes. If changes are needed, restart the flow: commit the fixes to `master` (keeping Project.toml at `X.Y.Z`), delete the old `release-X.Y.Z` branch and recreate it at the new master commit, and comment `@JuliaRegistrator register branch=release-X.Y.Z` on issue #124 again — Registrator then updates the registration to point at the new commit.
5. After TagBot has created the tag, the `release-X.Y.Z` branch is redundant (the tag protects the commit) and may be deleted.
6. When starting work on the next version, bump Project.toml to the next `-DEV` version and add a fresh `## Unreleased` section to CHANGELOG.md.

Example: v5.7.0 — release PR [#123](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/123), registration comment on [#124](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124#issuecomment-4966308410).
