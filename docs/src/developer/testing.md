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

1. Make a single commit titled `vX.Y.Z` that drops the `-DEV` suffix from
   `version` in `Project.toml` and renames the `## Unreleased` CHANGELOG
   section to `## X.Y.Z`. Land it on `master` (directly, or via a PR).
2. Push a frozen branch `release-X.Y.Z` pointing at the master commit that
   contains the release (the merge commit if it landed via PR):
   `git push origin <sha>:refs/heads/release-X.Y.Z`. Never commit to this
   branch — it pins the commit that gets registered even if `master` keeps
   moving.
3. Comment on the permanent (closed) registration issue
   [#124](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124):
   `@JuliaRegistrator register branch=release-X.Y.Z`, including release notes
   (markdown, typically the CHANGELOG entries for this version) in the same
   comment. Registrator ignores PR comments, and issue comments accept only a
   `branch=` target (no SHA), hence the pinned branch. The notes are added to
   the registry PR and to the GitHub release that TagBot creates. Registrator
   opens a PR in the General registry; once it merges, TagBot creates the
   `vX.Y.Z` tag at the registered commit and the GitHub release automatically.
4. Monitor the registration PR in General (linked in Registrator's reply to
   the triggering comment) until it merges — AutoMerge usually takes
   ~15–30 minutes. If AutoMerge fails or a registry maintainer requests
   changes, restart the flow: commit the fixes to `master` (keeping
   `Project.toml` at `X.Y.Z`), delete and recreate `release-X.Y.Z` at the new
   master commit, and comment on issue #124 again to re-trigger the
   registration pointing at the new branch. Once the registry PR has merged
   and TagBot has tagged, the `release-X.Y.Z` branch may be deleted.
5. Right after the release (registry PR merged, tag created), follow up with
   a new PR that bumps `Project.toml` to the next `-DEV` version (e.g.
   `5.7.1-DEV` after releasing `5.7.0`, or `5.8.0-DEV` if new features are
   anticipated) and adds a fresh `## Unreleased` section to `CHANGELOG.md`,
   so development commits never accumulate under a released version number.

The canonical, more detailed version of this procedure lives in the
`register-new-version` skill at
`.claude/skills/register-new-version/SKILL.md`; keep the two in sync.
