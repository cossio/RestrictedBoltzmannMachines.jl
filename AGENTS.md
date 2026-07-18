# AGENTS.md

This file provides repository guidance to Codex. Use the
`$restricted-boltzmann-machines-jl` skill for the package architecture,
commands, and subsystem-specific edit guidance.

## Repository workflow

- This is a Julia package supporting Julia 1.10 and later.
- Run commands from the repository root. Use `--project=.` for the package,
  `--project=test` for standalone test files, and `--project=docs` for docs.
- Run the narrowest relevant test file first, then
  `julia --project=. -e 'using Pkg; Pkg.test()'` when the change crosses
  subsystems or affects public behavior.
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

## GitHub operations

- A network-restricted sandbox can make `gh auth status` look like an invalid
  token. If it fails in the sandbox, retry it with host/network access before
  asking the user to reauthenticate; treat credentials as invalid only if that
  host-level check also fails.

## Changes and pull requests

- Add `CHANGELOG.md` entries only for user-facing package changes to source,
  APIs, behavior, or dependencies. Do not add entries for CI, workflows,
  agent plumbing, or other repository tooling.
- PRs get two automated reviews (Claude and Codex), both posting as the
  github-actions bot; Codex inline comments carry a `<!-- codex-review -->`
  marker and verdict bodies are prefixed with the reviewer's name. Treat
  approval from the latest run of each as the default handoff bar. Address
  each finding or explain the disagreement in its thread, reply to every
  thread, and leave thread resolution to the reviewer that opened it.
- A PR that changes `.github/workflows/claude-pr-review.yml` or
  `.github/workflows/codex-pr-review.yml` is reviewed by a fallback run
  dispatched from the default branch. If that dispatch fails, comment
  `@claude review this PR` (Claude) or dispatch the Codex workflow from the
  Actions tab with the PR number (Codex).
- Never merge a PR or enable auto-merge unless the repository owner explicitly
  instructs it.

## Releases

Use `$register-new-version` for release, registration, tagging, or publishing
tasks. The shared workflow is exposed to Codex at
`.agents/skills/register-new-version/SKILL.md`.
