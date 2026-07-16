# Architecture Reference

Read this file when you need the package map or repo-specific edit guidance.

## Workspace layout

- Root `Project.toml` defines a workspace with `test`, `docs`, `notebooks`, and
  `repl`.
- The root `Manifest.toml` is shared across the workspace members.
- Test-only and docs-only dependencies live in `test/Project.toml` and
  `docs/Project.toml`, not in the root package deps.

## Array and parameter invariants

- Layer dimensions come first and batch dimensions trail them. A layer of size
  `(N,)` accepts one vector sample or a batch shaped `(N, B)`; a Potts layer
  with `Q` classes and `N` sites uses `(Q, N)` or `(Q, N, B)`.
- `RBM.w` has shape `(size(visible)..., size(hidden)...)`. Preserve this
  convention for higher-dimensional layers rather than assuming matrices.
- `AbstractLayer{N}` records the number of layer dimensions. Every layer keeps
  all parameters in one `par` array whose first dimension selects the
  parameter and whose remaining dimensions are `size(layer)`, so
  `ndims(par) == N + 1`. Named fields such as `layer.θ` and `layer.γ` are views
  into `par`.
- The first Potts layer dimension indexes classes; the remaining layer
  dimensions index sites. Sampling and reductions must preserve that
  distinction while allowing trailing batch dimensions.
- Layer implementations provide `energy`, `cgfs`, `sample_from_inputs`,
  `mean_from_inputs`, `var_from_inputs`, and `mode_from_inputs` behavior.

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

- `test/runtests.jl` includes focused test files in independent modules, so a
  file can also run standalone with `--project=test`.
- Tests exercise multiple layer dimensions and use Zygote and
  FiniteDifferences for gradient checks.
- `test/jlarrays.jl` checks GPU-compatible semantics without physical hardware
  by using JLArrays with `allowscalar(false)`. Put device-generic coverage
  there; do not make CI depend on a physical GPU.
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
