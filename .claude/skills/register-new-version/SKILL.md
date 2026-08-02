---
name: register-new-version
description: Release a new version of RestrictedBoltzmannMachines.jl and register it in the Julia General registry, suggesting a ColPrac-compliant version number while leaving the final version decision to the user. Use when asked to release, register, tag, or publish a new package version.
---

# Releasing a new version

During development the version in Project.toml carries a `-DEV` suffix (e.g. `5.4.0-DEV`), and changes accumulate under an `## Unreleased` section in CHANGELOG.md.

## Choosing the version number (ColPrac semver)

Before anything else, analyze the changes and ask the user to choose the version. Never make the final version decision for the user.

1. Review the actual changes since the last registered version — the `## Unreleased` CHANGELOG entries **and** the commit history and diff since the previous release tag, which TagBot places at the last registered commit so the two reference points coincide (`git log vLAST..master`, `git diff vLAST..master`), since the CHANGELOG may be incomplete.
2. Cross-check the diff against CHANGELOG.md. The CHANGELOG covers only user-facing changes (source, APIs, behavior, dependencies — not CI, workflows, or repository tooling), so every user-facing change in the diff should have an `## Unreleased` entry, and every entry should match what the code actually does. Report any mismatch and propose the CHANGELOG fixes to the user, then stop: do not propose a version or create the release commit until the fixes are approved by the user, applied, and the cross-check passes — or the user explicitly decides no change is needed. The release notes are built from these entries, so a known mismatch must never ride into a release.
3. Classify every change using [ColPrac's extension of SemVer for Julia packages](https://docs.sciml.ai/ColPrac/stable/#Guidance-on-Package-Releases):
   - **Post-1.0:** bump major for breaking changes, minor for non-breaking features, and patch for bug fixes.
   - **Pre-1.0:** bump minor for breaking changes and patch for every non-breaking feature or bug fix.
   - Treat all documented APIs as public, including unexported names documented for normal use. Introducing a deprecation is non-breaking; removing one is breaking.
   - Treat dependency or Julia compatibility changes as non-breaking features, unless a dependency API exposed through this package makes the user-facing change breaking. Treat a compatibility change made solely to fix a bug as a bug fix.
   - Treat a correction to clearly broken behavior as a bug fix even when behavior changes incompatibly. Do not classify internal implementation changes, replacing an exception with non-error behavior, unspecified exception types or messages, floating-point details, new exports or supertypes, or textual representations as breaking solely for that reason.
4. Derive one suggested version from the highest bump required by the accumulated changes. Treat the `-DEV` version in Project.toml only as a hint — e.g. `5.6.1-DEV` may have accumulated features that suggest `5.7.0`.
5. Present the result in two parts: first a succinct summary of the user-facing changes since the last release, then the suggested version with a brief explanation identifying the changes that drive the bump. When a classification is genuinely borderline, also name the alternative version it would imply. Ask the user to confirm the suggestion or choose another version. Do not edit release files, commit, push, or trigger registration until the user explicitly chooses the final version.
6. If the **user** proposes a number, still perform the same review rather than accepting it blindly. If it conflicts with ColPrac, push back once with a brief explanation and ask them to confirm or revise. The user's decision is always final, including for borderline classifications.

## Procedure

1. **Release commit.** After the user explicitly chooses `X.Y.Z`, make a single commit titled `vX.Y.Z` that drops the `-DEV` suffix from `version` in Project.toml and renames the `## Unreleased` CHANGELOG section to `## X.Y.Z`. Land it on `master` (directly, or via a PR like [#123](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/123)).

2. **Trigger Registrator on the release commit.** Push the release commit and identify its exact SHA (the merge commit if the release landed via PR). Registration is triggered by posting this comment directly on that commit:

   ```markdown
   @JuliaRegistrator register

   Release notes:

   ## Breaking changes

   - blah
   ```

   Use the CHANGELOG entries for this version as the release notes. A commit comment pins registration to that commit, so no release branch or registration issue is needed. Post the comment through the commit-comments API in one of two modes:

   - **Post it yourself (preferred).** Run:

     ```bash
     gh api repos/cossio/RestrictedBoltzmannMachines.jl/commits/<sha>/comments -f body="$(cat <<'EOF'
     @JuliaRegistrator register

     Release notes:

     ## Breaking changes

     - blah
     EOF
     )"
     ```

     The quoted heredoc keeps quotes and backticks in the notes intact. Whether running the command or printing it for the user, emit every line flush-left: the indentation above is markdown list layout only — an indented `EOF` terminator never terminates the heredoc, and leading whitespace in the comment body can stop Registrator from recognizing the trigger. In a network-restricted sandbox `gh auth status` can look like an invalid token; retry with host/network access before concluding `gh` cannot post.

   - **Print the command for the user.** If `gh` cannot post the comment for whatever reason (missing binary, authentication or permission failure, sandbox restrictions, or a managed session whose proxy intercepts direct `api.github.com` calls), print the complete ready-to-run command above — real commit SHA and full release notes filled in, no placeholders left — so the user can copy it and run it themselves. A command substitution such as `$(gh pr view <N> --json mergeCommit -q .mergeCommit.oid)` is a placeholder in disguise: when the release lands via a PR, wait for the merge, resolve the merge-commit SHA, and only then print the command. Then wait for the user to confirm the comment was posted before monitoring the registration.

   Registrator replies on the commit with a link to the General registry PR; the notes flow into that PR and the GitHub release. Confirm the trigger worked by finding the new-version PR in [General](https://github.com/JuliaRegistries/General/pulls?q=RestrictedBoltzmannMachines) — the commit page can keep rendering a stale "0 comments" count after the comment posted, so it is not a reliable failure signal.

3. **Monitor the registry PR until it merges.** AutoMerge normally merges it within ~15–30 minutes. Watch for AutoMerge failures (version-increment, compat, or project-file checks) and comments from registry maintainers. If changes are needed, commit the fixes to `master` while keeping Project.toml at `X.Y.Z`, then post a new Registrator comment on the corrected commit. Registrator updates the registration to that commit. If the GitHub tooling in the session cannot read the General repo directly, read the public registry PR page.

4. **Tag and GitHub release.** Once the registry PR merges, TagBot creates the `vX.Y.Z` tag at the registered commit and the GitHub release with the notes automatically — no action needed.

5. **Start the next cycle — the release is unfinished until this PR exists.** Once the version is successfully registered in General (registry PR merged, tag created), immediately open the follow-up PR yourself; do not merely suggest it to the user or leave it for a later session, since a suggestion at the end of a long release run is easily dropped and leaves development commits accumulating under the released version number. The PR is a single commit (like [#185](https://github.com/cossio/RestrictedBoltzmannMachines.jl/pull/185)) that bumps Project.toml to the next `-DEV` version and adds a fresh, empty `## Unreleased` section at the top of CHANGELOG.md. Bump to the patch level `X.Y.(Z+1)-DEV` without asking — the `-DEV` number is only a hint, since the actual release number is re-derived from the changes when releasing (see the semver section above) — and mention in the PR that a minor or major hint (e.g. `X.(Y+1).0-DEV`) is available if the user anticipates features. Whenever release work resumes in a new session, verify this step happened: if `version` on `master` has no `-DEV` suffix and no bump PR is open, starting the next cycle is still pending and comes first.

A condensed human-facing copy of this procedure lives in `docs/src/developer/testing.md`; keep it in sync with this skill.

Worked example: [the v5.3.2 registration comment](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a4dcb8cee859c752881c6c1bb6051edaffcecf84#commitcomment-188488609).
