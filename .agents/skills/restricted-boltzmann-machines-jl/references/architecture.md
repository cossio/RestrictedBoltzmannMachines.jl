# Architecture Reference

Read this file when you need the package map or repo-specific edit guidance.

## Workspace layout

- Root `Project.toml` defines a workspace with `test`, `docs`, `notebooks`, and
  `repl`.
- The root `Manifest.toml` is shared across the workspace members.
- Test-only and docs-only dependencies live in `test/Project.toml` and
  `docs/Project.toml`, not in the root package deps.

## Code map

- `src/RestrictedBoltzmannMachines.jl`
  Main module, dependency imports, file inclusion order, and generic extension
  hooks such as `cpu`, `gpu`, `save_rbm`, and `load_rbm`.
- `src/layers/`
  Layer definitions and shared layer logic for binary, spin, Potts, Gaussian,
  ReLU-family, and related units.
- `src/rbm.jl` and `src/rbms/`
  Core RBM types plus convenience constructors and variants.
- `src/train/`
  Initialization, PCD, UCD, gradients, and minibatch iteration helpers
  (`src/train/infinite_minibatches.jl`).
- `src/gauge/`
  Gauge transforms such as zero-sum, hidden rescaling, and field shifting.
- `src/pseudolikelihood.jl`, `src/partition.jl`, `src/ais.jl`,
  `src/metropolis.jl`
  Evaluation, partition function, AIS, and sampling logic.
- `src/standardized.jl`, `src/centered.jl`
  Wrapper RBM forms with offsets and scaling.
- `ext/CUDAExt.jl`
  CUDA-backed `gpu` and `cpu` methods for arrays, layers, RBMs, gradients, and
  standardized RBMs.
- `ext/HDF5Ext.jl`
  HDF5 serialization with an explicit file-format version and layer-type tags.

## Repo conventions

- The package exports no names.
- Imports are explicit; preserve the existing style when adding dependencies.
- Optional dependency code belongs in `ext/`, not in the core package body.
- This repo leans on multi-dispatch and generic array code. Prefer extending
  existing abstractions over introducing type checks or backend-specific forks.

## Test and docs layout

- `test/runtests.jl` includes focused test files organized by subsystem.
- `test/aqua.jl` is part of the suite, so package-health regressions can break
  CI.
- `docs/make.jl` removes generated markdown under `docs/src/literate/`, runs
  Literate on the `.jl` sources, builds Documenter pages, then removes the
  generated markdown again.

## Change heuristics

- New layer type:
  add the layer definition, wire shared behavior where needed, add focused
  tests, and update docs/examples. If the layer should support CUDA or HDF5,
  extend the relevant files in `ext/`.
- New training or evaluation API:
  add core implementation, tests, and user-facing docs together.
- New optional integration:
  use `[weakdeps]` and `[extensions]`, keep the base API generic, and add
  extension-specific tests if feasible.
- File-format changes:
  treat `ext/HDF5Ext.jl` as compatibility-sensitive and update the format
  version intentionally.
