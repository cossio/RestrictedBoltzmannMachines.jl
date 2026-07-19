# Repository instructions

This file provides guidance to coding agents working in this repository.

## Project overview

RestrictedBoltzmannMachines.jl is a Julia package for training and inference
with Restricted Boltzmann Machines (RBMs). It supports multiple layer types
(Binary, Spin, Potts, Gaussian, and ReLU variants), GPU acceleration through
CUDA.jl, and HDF5 persistence. It requires Julia 1.10 or later.

## Repository workflow

- This is a Julia package supporting Julia 1.10 and later.
- Run commands from the repository root. Use `--project=.` for the package,
  `--project=test` for standalone test files, and `--project=docs` for docs.
- Run the narrowest relevant test file first, then
  `julia --project=. -e 'using Pkg; Pkg.test()'` when the change crosses
  subsystems or affects public behavior.
- Load the package with
  `julia --project=. -e 'import RestrictedBoltzmannMachines as RBMs'`.
- Common commands:

  ```bash
  # Run all tests
  julia --project=. -e 'using Pkg; Pkg.test()'

  # Run one test file with test-only dependencies
  julia --project=test test/pcd.jl

  # Start a package REPL, then import the package
  julia --project=.
  ```

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
  can run standalone with `--project=test`; the suite uses property-based tests
  across dimensions and gradient checks with Zygote and FiniteDifferences.

## Package architecture and invariants

- The package exports no symbols. Prefer
  `import RestrictedBoltzmannMachines as RBMs` or explicit
  `using RestrictedBoltzmannMachines: ...`.
- Layer dimensions come first and trailing dimensions are batch dimensions. A
  layer of size `(N,)` accepts `(N,)` or `(N, B)` data; a Potts layer with `Q`
  classes and `N` sites accepts `(Q, N)` or `(Q, N, B)`.
- `RBM.w` has shape `(size(visible)..., size(hidden)...)`; preserve this
  convention for higher-dimensional layers rather than assuming matrices.
- `AbstractLayer{N}` records the number of layer dimensions. Layer parameters
  share one `par` array whose first dimension selects the parameter and whose
  remaining dimensions are `size(layer)`, so `ndims(par) == N + 1`. Named
  properties such as `layer.θ` and `layer.γ` are views into `par`.
- Each layer implements `energy`, `cgfs`, `sample_from_inputs`,
  `mean_from_inputs`, `var_from_inputs`, and `mode_from_inputs`.
- Binary, Spin, and Potts layers have one parameter (`θ`); Gaussian layers have
  two (`θ` and `γ`); dReLU layers have four.
- For Potts layers, the first layer dimension indexes classes and the remaining
  layer dimensions index sites. Sampling and reductions must preserve this
  distinction while allowing trailing batch dimensions. Thus a Potts layer
  with spatial shape `N...` has size `(Q, N...)`, while `par` has shape
  `(1, Q, N...)`.
- Layer types include `Binary`, `Spin`, `Potts`, `Gaussian`, `ReLU`, `dReLU`,
  `pReLU`, `xReLU`, `nsReLU`, and `PottsGumbel`.
- `RBM{V,H,W}` stores the `visible` layer, `hidden` layer, and weights `w`.
  `CenteredRBM` adds offset parameters; `StandardizedRBM` adds offsets and
  scales.
- Preserve generic array and multiple-dispatch behavior in core code. Put
  dependency-specific methods in `ext/`; treat the versioned HDF5 format in
  `ext/HDF5Ext.jl` as compatibility-sensitive.
- Literate sources live in `docs/src/literate/`. Generated Markdown there is a
  transient build artifact removed by `docs/make.jl`.

## Module organization

- `src/layers/` contains layer definitions. `abstractlayer.jl` defines the
  interface, individual files implement layer types, and `common.jl` contains
  shared utilities.
- `src/rbms/` contains convenience constructors such as `BinaryRBM` and
  `HopfieldRBM`.
- `src/train/` contains training code: persistent contrastive divergence in
  `pcd.jl`, data-driven initialization in `initialization.jl`, and gradients in
  `gradient.jl`.
- `src/gauge/` contains gauge transformations, including `zerosum.jl`,
  `rescale_hidden.jl`, and `shift_fields.jl`.
- `src/util/` contains linear-algebra helpers, one-hot encoding, and truncated
  normal sampling.
- `ext/` contains the CUDA and HDF5 package extensions.

## Key functions

- Training: `pcd!` and `initialize!`.
- Sampling: `sample_v_from_h`, `sample_h_from_v`, `sample_v_from_v`, and
  `metropolis`.
- Evaluation: `free_energy`, `energy`, `log_pseudolikelihood`,
  `log_partition`, and `reconstruction_error`.
- Partition-function estimation: `aise` (annealed importance sampling) and
  `raise` (reverse AIS).

## GitHub operations

- A network-restricted sandbox can make `gh auth status` look like an invalid
  token. If it fails in the sandbox, retry it with host/network access before
  asking the user to reauthenticate; treat credentials as invalid only if that
  host-level check also fails.

## Changes and pull requests

- Add `CHANGELOG.md` entries only for user-facing package changes to source,
  APIs, behavior, or dependencies. Do not add entries for CI, workflows,
  agent plumbing, or other repository tooling.
- PRs receive automated review comments from Codex Cloud and the default Claude
  App workflow in `.github/workflows/claude-code-review.yml`. Address each
  actionable finding or explain the disagreement in its thread, reply to every
  thread, and resolve it once addressed.
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
