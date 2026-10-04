# Repository instructions

This file provides guidance to coding agents working in this repository.

## Project overview

RestrictedBoltzmannMachines.jl is a Julia package for training and inference
with Restricted Boltzmann Machines (RBMs). It supports multiple layer types
(Binary, Spin, Potts, Gaussian, and ReLU variants), GPU acceleration through
CUDA.jl, and HDF5 persistence. It requires Julia 1.12 or later.

## Repository workflow

- Run commands from the repository root. Use `--project=.` for the package,
  `--project=test` for standalone test files, and `--project=docs` for docs.
- Run the narrowest relevant test file first, then
  `julia --project=. -e 'using Pkg; Pkg.test()'` when the change crosses
  subsystems or affects public behavior.
- Do not wait for the full test suite to pass locally before opening a PR.
  GitHub CI runs the complete suite on every PR at no cost, so focus local
  runs on a few tests targeting the change and open the PR a bit earlier;
  let CI provide full coverage.
- Load the package with
  `julia --project=. -e 'import RestrictedBoltzmannMachines as RBMs'`.

## Workspace and environment

- The root Julia workspace includes `test`, `docs`, `notebooks`, and `repl`
  and uses one shared, gitignored root `Manifest.toml`.
- The test project needs the root package developed into it. If that setup is
  missing, run
  `julia --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`.
- No external services are required. CUDA and HDF5 are optional package
  extensions. The test project includes HDF5 coverage; CPU tests do not need
  CUDA or physical GPU hardware.
- Test GPU semantics in `test/jlarrays.jl` with JLArrays and
  `allowscalar(false)`. Do not commit tests that require physical GPU hardware.
- Tests in `test/runtests.jl` are organized as independent modules. Test files
  can run standalone with `--project=test`.

## Package architecture and invariants

- The package exports no symbols. Prefer
  `import RestrictedBoltzmannMachines as RBMs` or explicit
  `using RestrictedBoltzmannMachines: ...`.
- Layer dimensions come first and trailing dimensions are batch dimensions. A
  layer of size `(N,)` accepts `(N,)` or `(N, B)` data. For Potts layers the
  first layer dimension indexes the `Q` classes and the remaining layer
  dimensions index sites: a Potts layer with spatial shape `N...` has size
  `(Q, N...)`, `par` has shape `(1, Q, N...)`, and sampling and reductions
  must preserve this distinction while allowing trailing batch dimensions.
- `RBM.w` has shape `(size(visible)..., size(hidden)...)`; preserve this
  convention for higher-dimensional layers rather than assuming matrices.
- `AbstractLayer{N}` records the number of layer dimensions. Layer parameters
  share one `par` array whose first dimension selects the parameter and whose
  remaining dimensions are `size(layer)`, so `ndims(par) == N + 1`. Named
  properties such as `layer.θ` and `layer.γ` are views into `par`.
- Each layer implements `energy`, `cgfs`, `sample_from_inputs`,
  `mean_from_inputs`, `var_from_inputs`, and `mode_from_inputs`, plus the
  moments interface: `moments_from_samples`, `moments_from_inputs`, and
  `∂energy_from_moments`, which share one canonical moments-array layout
  (first axis = moment index, then `size(layer)`, then batch dimensions).
- Binary, Spin, and Potts layers have one parameter (`θ`); Gaussian and ReLU
  layers have two (`θ` and `γ`); dReLU, pReLU, and xReLU have four; nsReLU has
  three.
- `RBM{V,H,W}` stores the `visible` layer, `hidden` layer, and weights `w`.
  `StandardizedRBM` adds offsets and scales. `CenteredRBM` is an alias for a
  `StandardizedRBM` whose scales are lazy `FillArrays.Trues`, which are
  immutable: in-place updates of a centered model change only its offsets.
- Preserve generic array and multiple-dispatch behavior in core code. Put
  dependency-specific methods in `ext/`; treat the versioned HDF5 format in
  `ext/HDF5Ext.jl` as compatibility-sensitive.
- Literate sources live in `docs/src/literate/`. Generated Markdown there is a
  transient build artifact removed by `docs/make.jl`.

## GitHub operations

- A network-restricted sandbox can make `gh auth status` look like an invalid
  token. If it fails in the sandbox, retry it with host/network access before
  asking the user to reauthenticate; treat credentials as invalid only if that
  host-level check also fails.

## Changes and pull requests

- Add `CHANGELOG.md` entries under `## Unreleased` only for changes a user
  observes through the `public`/`export`ed API (names, signatures, behavior,
  results) or through performance, dependencies, or supported Julia versions.
  Not for internal refactors, tests, docs, formatting, CI, workflows, or other
  repository tooling: touching `src/` does not by itself warrant an entry.
- PR reviews are not automatic, and requesting one is not your call: the
  repository owner triggers a Claude or Codex Cloud review when they want
  one. Never trigger a review yourself, and do not ask for one. When review
  comments do arrive, address each actionable finding or explain the
  disagreement in its thread, reply to every thread, and resolve it once
  addressed.
- Follow `REVIEW.md`; flag substantial avoidable complexity only when a
  materially simpler design satisfies the current requirements.
- Never merge a PR or enable auto-merge unless the repository owner explicitly
  instructs it.

## Releases

- During development, the version in `Project.toml` carries a `-DEV` suffix
  (for example, `5.4.0-DEV`), and changes accumulate under `## Unreleased` in
  `CHANGELOG.md`.
- Choose release numbers using ColPrac's Julia package SemVer guidance. For this
  post-1.0 package, breaking changes bump major, non-breaking features bump
  minor, and bug fixes bump patch. Suggest one version with a brief explanation,
  but always leave the final decision to the user and wait for explicit
  confirmation before making release changes.
- Use `$register-new-version` for release, registration, tagging, or publishing
  tasks. The shared workflow lives at
  `.claude/skills/register-new-version/SKILL.md` and is exposed to Codex at
  `.agents/skills/register-new-version/SKILL.md`. It covers the release commit,
  triggering Registrator directly on that commit, and monitoring the General
  registry PR.
