---
name: register-new-version
description: Release a new version of RestrictedBoltzmannMachines.jl and register it in the Julia General registry, suggesting a ColPrac-compliant version number while leaving the final version decision to the user. Use when asked to release, register, tag, or publish a new package version.
---

# Releasing a new version

During development the version in Project.toml carries a `-DEV` suffix (e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in CHANGELOG.md.

## Choosing the version number (ColPrac semver)

Before anything else, analyze the changes and ask the user to choose the version. Never make the final version decision for the user.

1. Review the actual changes since the last registered version — the `## Unreleased` CHANGELOG entries **and** the commit history/diff since the previous release tag (`git log vLAST..master`), since the CHANGELOG may be incomplete.
2. Classify every change using [ColPrac's extension of SemVer for Julia packages](https://docs.sciml.ai/ColPrac/stable/#Guidance-on-Package-Releases):
   - **Post-1.0:** bump major for breaking changes, minor for non-breaking features, and patch for bug fixes.
   - **Pre-1.0:** bump minor for breaking changes and patch for every non-breaking feature or bug fix.
   - Treat all documented APIs as public, including unexported names documented for normal use. Introducing a deprecation is non-breaking; removing one is breaking.
   - Treat dependency or Julia compatibility changes as non-breaking features, unless a dependency API exposed through this package makes the user-facing change breaking. Treat a compatibility change made solely to fix a bug as a bug fix.
   - Treat a correction to clearly broken behavior as a bug fix even when behavior changes incompatibly. Do not classify internal implementation changes, replacing an exception with non-error behavior, unspecified exception types or messages, floating-point details, new exports or supertypes, or textual representations as breaking solely for that reason.
3. Derive one suggested version from the highest bump required by the accumulated changes. Treat the `-DEV` version in Project.toml only as a hint — e.g. `5.6.1-DEV` may have accumulated features that suggest `5.7.0`.
4. Present the suggestion with a brief explanation identifying the changes that drive the bump, then ask the user to confirm it or choose another version. Do not edit release files, commit, push, or trigger registration until the user explicitly chooses the final version.
5. If the **user** proposes a number, still perform the same review rather than accepting it blindly. If it conflicts with ColPrac, push back once with a brief explanation and ask them to confirm or revise. The user's decision is always final, including for borderline classifications.

## Procedure

1. **Release commit.** After the user explicitly chooses `X.Y.Z`, make a single commit titled `vX.Y.Z` that drops the `-DEV` suffix from `version` in Project.toml and renames the `## Unreleased` CHANGELOG section to `## X.Y.Z`. Land it on `master` (directly, or via a PR like [#123](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/123)).

2. **Trigger Registrator on the release commit.** Push the release commit, open that exact commit on GitHub (the merge commit if the release landed via PR), and post this comment directly on it:

   ```markdown
   @JuliaRegistrator register

   Release notes:

   ## Breaking changes

   - blah
   ```

   Use the CHANGELOG entries for this version as the release notes. A commit comment pins registration to that commit, so no release branch or registration issue is needed. Post through the GitHub commit page or the commit-comments API (`gh api repos/cossio/RestrictedBoltzmannMachines.jl/commits/<sha>/comments -f body="..."`). Registrator replies on the commit with a link to the General registry PR; the notes flow into that PR and the GitHub release.

3. **Monitor the registry PR until it merges.** AutoMerge normally merges it within ~15–30 minutes. Watch for AutoMerge failures (version-increment, compat, or project-file checks) and comments from registry maintainers. If changes are needed, commit the fixes to `master` while keeping Project.toml at `X.Y.Z`, then post a new Registrator comment on the corrected commit. Registrator updates the registration to that commit. If the GitHub tooling in the session cannot read the General repo directly, read the public registry PR page.

4. **Tag and GitHub release.** Once the registry PR merges, TagBot creates the `vX.Y.Z` tag at the registered commit and the GitHub release with the notes automatically — no action needed.

5. **Start the next cycle.** Once the version is successfully registered in General (registry PR merged, tag created), always suggest that the user follow up with a new PR that bumps Project.toml to the next `-DEV` version and adds a fresh `## Unreleased` section to CHANGELOG.md, so development commits never accumulate under a released version number. (E.g. after releasing `5.7.0`, bump to `5.7.1-DEV`, or `5.8.0-DEV` if new features are anticipated — the choice is only a hint, since the actual release number is re-derived from the changes when releasing; see the semver section above.)

A condensed human-facing copy of this procedure lives in `docs/src/developer/testing.md`; keep it in sync with this skill.

Worked example: [the v5.3.2 registration comment](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a4dcb8cee859c752881c6c1bb6051edaffcecf84#commitcomment-188488609).
