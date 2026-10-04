# Testing and releasing

## Running tests

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file (uses the test/ project for test-only deps like Zygote, QuadGK)
julia --project=test test/pcd.jl
```

Tests in `test/runtests.jl` are organized as independent modules (each file is
wrapped in its own `module`), so each test file can also be run standalone.
Tests use property-based checks across dimensions, and gradient checking with
Zygote.

## GPU testing without a GPU

GitHub CI has no GPU. GPU compatibility is tested in `test/jlarrays.jl` using
[JLArrays](https://github.com/JuliaGPU/GPUArrays.jl) with
`allowscalar(false)`, which catches scalar-indexing code paths that would fail
on a real GPU. Do not commit tests that require physical GPU hardware; run
those locally only.

## Releasing a new version

During development the version in `Project.toml` carries a `-DEV` suffix
(e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in
`CHANGELOG.md`. Release numbers follow
[ColPrac's Julia package guidance](https://docs.sciml.ai/ColPrac/stable/#Guidance-on-Package-Releases):
for this post-1.0 package, a major bump for breaking changes, a minor bump for
non-breaking features, and a patch bump for bug fixes.

A release is one commit `vX.Y.Z` on `master` that drops the `-DEV` suffix and
renames `## Unreleased` to `## X.Y.Z`, followed by a `@JuliaRegistrator register`
comment on that commit (with the CHANGELOG entries under `Release notes:`).
Once the General registry PR merges, TagBot tags the commit and creates the
GitHub release, and a follow-up PR bumps `Project.toml` to the next `-DEV`
version with a fresh `## Unreleased` section. Post the trigger phrase only in
that commit comment: the bot matches it in any issue or PR comment, even inside
backticks.

The authoritative step-by-step procedure lives in the `register-new-version`
skill (`.claude/skills/register-new-version/SKILL.md`, mirrored for Codex at
`.agents/skills/register-new-version/SKILL.md`).
