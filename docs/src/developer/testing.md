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
`CHANGELOG.md`. Choose the release number using
[ColPrac's Julia package guidance](https://docs.sciml.ai/ColPrac/stable/#Guidance-on-Package-Releases):
for this post-1.0 package, use a major bump for breaking changes, a minor bump
for non-breaking features (including ordinary dependency or Julia compatibility
changes), and a patch bump for bug fixes (including compatibility changes made
solely to fix a bug). Treat documented unexported names as public API,
corrections to clearly broken behavior as bug fixes, introducing deprecations
as non-breaking, and removing deprecations as breaking. The release agent
suggests one version with a brief explanation, but the user always makes the
final decision. Do not change release files or begin registration until the
user explicitly chooses the version. Then:

1. Make a single commit titled `vX.Y.Z` that drops the `-DEV` suffix from
   `version` in `Project.toml` and renames the `## Unreleased` CHANGELOG
   section to `## X.Y.Z`. Land it on `master` (directly, or via a PR).
2. Push the release commit, open that exact commit on GitHub (the merge commit
   if the release landed via PR), and comment `@JuliaRegistrator register`
   directly on it. Include a `Release notes:` section containing the CHANGELOG
   entries for this version in the same comment. The commit comment pins the
   registered SHA, so no release branch is needed. Registrator replies on the
   commit with the General registry PR.
3. Monitor the registration PR until it merges — AutoMerge usually takes
   ~15–30 minutes. If AutoMerge fails or a registry maintainer requests
   changes, commit the fixes to `master` while keeping `Project.toml` at
   `X.Y.Z`, then post a new Registrator comment on the corrected commit.
4. Once the registry PR merges, TagBot creates the `vX.Y.Z` tag at the
   registered commit and the GitHub release with the supplied notes
   automatically.
5. Right after the release (registry PR merged, tag created), follow up with
   a new PR that bumps `Project.toml` to the next `-DEV` version (e.g.
   `5.7.1-DEV` after releasing `5.7.0`, or `5.8.0-DEV` if new features are
   anticipated) and adds a fresh `## Unreleased` section to `CHANGELOG.md`,
   so development commits never accumulate under a released version number.

The canonical, more detailed version of this procedure lives in the shared
`register-new-version` skill. It is available to Claude at
`.claude/skills/register-new-version/SKILL.md` and to Codex at
`.agents/skills/register-new-version/SKILL.md`; keep this page in sync with the
skill.
