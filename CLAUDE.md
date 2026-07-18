# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RestrictedBoltzmannMachines.jl is a Julia package for training and inference with Restricted Boltzmann Machines (RBMs). It supports multiple layer types (Binary, Spin, Potts, Gaussian, ReLU variants), GPU acceleration via CUDA.jl, and HDF5 persistence. Requires Julia 1.10+.

## Common Commands

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file (uses test/ project for test-only deps like Zygote, QuadGK)
julia --project=test test/pcd.jl

# Start a REPL with the package loaded
julia --project=.
julia> import RestrictedBoltzmannMachines as RBMs
```

## Architecture

### Array Dimension Convention

Data arrays have layer dimensions first and batch dimension last. For a layer of size `(N,)`, a single sample is a vector of length `N`, and a batch is a matrix `(N, B)`. For Potts layers of size `(Q, N)`, a batch is `(Q, N, B)`. The weight matrix `w` has shape `(size(visible)..., size(hidden)...)`. Functions like `energy`, `free_energy`, `sample_from_inputs` all follow this convention and broadcast over the trailing batch dimension.

### Type Hierarchy

`AbstractLayer{N}` (N = ndims) is the base type for all unit types. Each layer implements: `energy`, `cgfs`, `sample_from_inputs`, `mean_from_inputs`, `var_from_inputs`, `mode_from_inputs`.

All layers store parameters in a single `.par` array. The first dimension is the number of parameters of the layer type (e.g. 1 for Binary/Spin/Potts which have only `θ`, 2 for Gaussian which has `θ` and `γ`, 4 for dReLU). The remaining dimensions are the layer's spatial dimensions (the grid of units). So `ndims(par) == N + 1` where `N` is the layer ndims. Named parameter accessors (e.g. `layer.θ`, `layer.γ`) are views into `.par`.

Potts is special: its first layer dimension is the one-hot (categorical) dimension with `Q` classes, and the remaining dimensions describe the grid of units. So `size(potts_layer) == (Q, N...)` and `par` has shape `(1, Q, N...)`.

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

## Pull Requests

Every PR receives two automated reviews: a Claude review (`.github/workflows/claude-pr-review.yml`) and a Codex review (`.github/workflows/codex-pr-review.yml`). Both post inline comments and end with a formal review verdict — approve, request changes, or comment — submitted through the shared github-actions bot account: Codex verdict bodies begin with "Codex automated review" and its inline comments carry a `<!-- codex-review -->` marker, while Claude prefixes its verdicts with "Claude review:". The bot account's visible review state reflects whichever reviewer submitted an approve/request-changes verdict most recently. Each workflow keeps a PR from changing the rules applied to itself: the Claude action self-skips when its workflow file differs from the default branch, and CI then dispatches a fallback review run from the default-branch version via `workflow_dispatch` (a CI-posted comment explains this; if the dispatch fails, comment `@claude review this PR`), while the Codex workflow runs on `pull_request_target`, so the default-branch version always executes and reads REVIEW.md from the base revision. The hosted Codex reviewer (`chatgpt-codex-connector`) may additionally post advisory comment reviews; treat its findings like any other reviewer feedback. When authoring a PR, treat both workflow reviewers' approval as the default bar before merge: address their findings by pushing fixes, or reply in the review threads with reasoning if a finding is mistaken — each reviewer re-runs on every push, and a fresh verdict supersedes that reviewer's previous one.

Reply in every review thread on your PR before considering the work done: either point at the commit that addresses the finding, or explain why it is mistaken. Do not resolve review threads yourself — the reviewer resolves a thread once convinced (by the fix or by your reply), and replies back when not.

Never merge a PR or enable auto-merge. Merging happens only when the repo owner clicks merge on GitHub or explicitly instructs it.

## Releasing a new version

During development the version in Project.toml carries a `-DEV` suffix (e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in CHANGELOG.md. CHANGELOG.md records only changes to the package code that affect users (source, API, behavior, dependencies). Do not add entries for CI/workflow changes, repo tooling, or other development-infrastructure changes. To release and register a new version, use the shared `register-new-version` skill (`.claude/skills/register-new-version/SKILL.md`, exposed to Codex through `.agents/skills/register-new-version/SKILL.md`), which documents the full workflow: release commit, frozen `release-X.Y.Z` registration branch, triggering Registrator from issue [#124](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124), and monitoring the General registry PR.
