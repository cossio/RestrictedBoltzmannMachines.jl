---
name: restricted-boltzmann-machines-jl
description: >
  Use for tasks specific to this RestrictedBoltzmannMachines.jl repository:
  editing RBM layer/model internals, training or sampling code, gauge
  transforms, centered or standardized RBMs, CUDA/HDF5 extensions, the Julia
  workspace setup, package tests, docs, literate examples, notebooks, and repo
  maintenance. Do not use for generic Julia questions or unrelated repositories
  unless the task explicitly depends on this repo's files, APIs, structure, or
  conventions.
---

# RestrictedBoltzmannMachines.jl Repo Skill

Use this skill when working inside this repository. It is repo-specific
guidance layered on top of general Julia knowledge.

## Repo snapshot

- Root package `RestrictedBoltzmannMachines` uses a Julia workspace with
  `test`, `docs`, `notebooks`, and `repl` subprojects under one shared
  `Manifest.toml`.
- The package exports no symbols. In examples, prefer
  `import RestrictedBoltzmannMachines as RBMs`, or use explicit
  `using RestrictedBoltzmannMachines: ...`.
- The main module is assembled from focused source files under `src/`:
  `layers/`, `rbms/`, `train/`, `gauge/`, utilities, partition/AIS, and the
  centered/standardized wrappers.
- Optional integrations live in `ext/CUDAExt.jl` and `ext/HDF5Ext.jl`.

## Working rules

1. Match the repository's import style. Prefer `import` or explicit
   `using ...: ...`; do not introduce bare `using RestrictedBoltzmannMachines`.
2. Respect the root workspace. Run commands from the repo root with
   `--project=.`, `--project=test`, or `--project=docs` as appropriate.
3. If you change public behavior or math-sensitive code, update focused tests in
   `test/` in the same change.
4. If you change user-facing APIs or examples, update the relevant docs surface:
   `README.md`, `docs/src`, or `docs/src/literate`.
5. Keep dependency-specific methods inside `ext/`; keep only generic
   declarations or stubs in `src/RestrictedBoltzmannMachines.jl`.
6. Preserve the existing layer families and wrappers unless the change is
   explicitly about expanding them: `Binary`, `Spin`, `Potts`, `Gaussian`,
   `ReLU`, `dReLU`, `pReLU`, `xReLU`, `nsReLU`, `PottsGumbel`,
   `StandardizedRBM`, and `CenteredRBM`.
7. Be careful with shapes, element types, and array backends. The package is
   designed around generic arrays, with optional GPU support through CUDA.

## Read these references as needed

- `references/architecture.md` for the package map and edit heuristics.
- `references/commands.md` for common local commands.

## Task guidance

### Core model or algorithm changes

- Inspect the closest subsystem first: `src/layers/`, `src/rbm.jl`,
  `src/gradient.jl`, `src/train/`, `src/partition.jl`, `src/ais.jl`,
  `src/metropolis.jl`, `src/standardized.jl`, or `src/centered.jl`.
- Search `test/` for the same concept before editing. This repo's tests are
  already decomposed by feature.
- Preserve multi-dispatch across layer types and array backends. Avoid
  accidentally hard-coding CPU-only behavior into core methods.

### Optional integration work

- Prefer Julia package extensions via `[weakdeps]` and `[extensions]`.
- If an API hook is needed, declare the generic function in
  `src/RestrictedBoltzmannMachines.jl` and put dependency-specific methods in
  `ext/YourExt.jl`.
- Mirror the existing CUDA and HDF5 extension structure.

### Docs and examples

- Literate source files live under `docs/src/literate/`. Generated `.md` files
  are transient build artifacts and are deleted by `docs/make.jl`.
- Keep README snippets aligned with real constructors and training APIs.
- Treat notebooks and `repl/` scripts as exploratory or demo material, not the
  authoritative API reference.

## Validation

- Run the narrowest relevant test file first, then the broader suite if the
  change crosses subsystems.
- Run `julia --project=test test/jlarrays.jl` when changing generic array or
  backend behavior; it checks GPU semantics without physical GPU hardware.
- Build docs when touching examples, docstrings, or `docs/src/literate`.
- If `Project.toml` changes, confirm the workspace still resolves and the root
  manifest remains coherent.
