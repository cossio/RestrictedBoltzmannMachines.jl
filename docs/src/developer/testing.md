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
Zygote and FiniteDifferences.

## GPU testing without a GPU

GitHub CI has no GPU. GPU compatibility is tested in `test/jlarrays.jl` using
[JLArrays](https://github.com/JuliaGPU/GPUArrays.jl) with
`allowscalar(false)`, which catches scalar-indexing code paths that would fail
on a real GPU. Do not commit tests that require physical GPU hardware; run
those locally only.

## Releasing a new version

During development the version in `Project.toml` carries a `-DEV` suffix
(e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in
`CHANGELOG.md`. To release:

1. On `master`, make a single commit titled `vX.Y.Z` that drops the `-DEV`
   suffix from `version` in `Project.toml` and renames the `## Unreleased`
   CHANGELOG section to `## X.Y.Z`. Push it.
2. Comment `@JuliaRegistrator register` on that commit on GitHub, including
   release notes (markdown, typically the CHANGELOG entries for this version)
   in the same comment. The notes are added to the registry PR and to the
   GitHub release that TagBot creates. Registrator opens a PR in the General
   registry; once it merges, TagBot creates the `vX.Y.Z` tag and GitHub
   release automatically.
3. When starting work on the next version, bump `Project.toml` to the next
   `-DEV` version and add a fresh `## Unreleased` section to `CHANGELOG.md`.
