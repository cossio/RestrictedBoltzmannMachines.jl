---
name: register-new-version
description: Release a new version of RestrictedBoltzmannMachines.jl and register it in the Julia General registry, suggesting a semver-correct version number from the changes since the last release. Use when asked to release, register, tag, or publish a new package version.
---

# Releasing a new version

During development the version in Project.toml carries a `-DEV` suffix (e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in CHANGELOG.md.

## Choosing the version number (semver)

Before anything else, determine the right version number and confirm it with the user:

1. Review the actual changes since the last registered version — the `## Unreleased` CHANGELOG entries **and** the commit history/diff since the previous release tag (`git log vLAST..master`), since the CHANGELOG may be incomplete. Classify them:
   - **Breaking** (removed/renamed exported names, changed signatures, semantics or return types of existing public API, dropped Julia/package compat) → bump **major**.
   - **New features** (new exported functions/types/methods, new keyword options, new capabilities — even small ones) → bump **minor**.
   - **Bug fixes / internal changes only** (fixes, performance, docs, tests, CI) → bump **patch**.
   Note the `-DEV` version in Project.toml is only a hint of what was anticipated, not authoritative — e.g. `5.6.1-DEV` may still have accumulated features that warrant `5.7.0`.
2. Suggest the resulting number to the user, with a one-line justification citing the entries that drive the bump, and get their confirmation.
3. If the **user** proposes a number, still perform the same review rather than accepting it: check it against the changes, and if it doesn't follow semver (e.g. a patch bump when exported behavior changed, or a major bump for pure fixes), say so, explain which changes conflict with it, and ask them to confirm or revise. Only proceed once the user has confirmed a number after seeing your assessment. (A borderline call — e.g. whether a bug-fix-driven behavior change counts as breaking — is the user's to make; your job is to surface it.)

## Procedure

1. **Release commit.** Make a single commit titled `vX.Y.Z` that drops the `-DEV` suffix from `version` in Project.toml and renames the `## Unreleased` CHANGELOG section to `## X.Y.Z`. Land it on `master` (directly, or via a PR like [#123](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/123)).

2. **Frozen registration branch.** Create a branch `release-X.Y.Z` pointing at the master commit that contains the release (the merge commit if it landed via PR) and push it: `git push origin <sha>:refs/heads/release-X.Y.Z`. Never commit to this branch — its whole point is that its HEAD *is* the release commit, so registration targets the exact commit even if master keeps moving in the meantime.

3. **Trigger Registrator.** Comment on issue [#124 "Julia registration"](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124) — the single permanent issue used for all registrations, kept closed on purpose so it doesn't pollute the issue list (comments on closed issues still trigger Registrator; do NOT open a new issue per release). Include release notes (markdown, typically the CHANGELOG entries for this version) in the same comment:

   ```
   @JuliaRegistrator register branch=release-X.Y.Z

   Release notes:

   ## Breaking changes

   - blah
   ```

   Caveats learned the hard way: Registrator ignores PR comments entirely ("disabled", it replies); and from an issue comment only `branch=<name>` is accepted — there is no way to pin a SHA, which is why step 2 exists. Registrator resolves the branch to a concrete SHA when it processes the comment and replies with a link to the PR it opens in the General registry. The notes are added to that registry PR and to the GitHub release. (If the register comment was posted without notes, re-invoke Registrator with them to update the registration — see [Registrator's reply](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a4dcb8cee859c752881c6c1bb6051edaffcecf84#commitcomment-188488609).)

4. **Monitor the registry PR until it merges.** Registrator's reply to the triggering comment links it (e.g. [this reply](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124#issuecomment-4966310598) → [JuliaRegistries/General#161274](https://github.com/JuliaRegistries/General/pull/161274) for v5.7.0). AutoMerge normally merges it within ~15–30 minutes. Watch for AutoMerge failures (version-increment, compat or project-file checks) and for comments from registry maintainers requesting changes. If changes are needed, restart the flow: commit the fixes to `master` (keeping Project.toml at `X.Y.Z`), delete the old `release-X.Y.Z` branch and recreate it at the new master commit, and comment `@JuliaRegistrator register branch=release-X.Y.Z` on issue #124 again — Registrator then updates the registration to point at the new commit. (If the GitHub tooling in the session cannot read the General repo directly, the registry PR page is public and can be read via web fetch.)

5. **Tag and GitHub release.** Once the registry PR merges, TagBot creates the `vX.Y.Z` tag at the registered commit and the GitHub release (with the notes) automatically — no action needed. Afterwards the `release-X.Y.Z` branch is redundant (the tag protects the commit) and may be deleted.

6. **Start the next cycle.** When starting work on the next version, bump Project.toml to the next `-DEV` version and add a fresh `## Unreleased` section to CHANGELOG.md.

A condensed human-facing copy of this procedure lives in `docs/src/developer/testing.md`; keep it in sync with this skill.

Worked example: v5.7.0 — release PR [#123](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/123), registration comment on [#124](https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/124#issuecomment-4966308410).
