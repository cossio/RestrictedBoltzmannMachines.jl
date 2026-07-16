# Review instructions

This is a Julia package for training and inference with Restricted Boltzmann
Machines. Prioritize numerical and statistical correctness above everything
else: a subtly wrong formula is worse than a crash, because tests may not
catch it and users get silently wrong results.

Report real problems only. Do not comment on style, formatting, or
preferences; do not restate what the diff does. If you find nothing
significant, post no findings.

## What Important means here

Reserve Important (blocking-severity) for findings that would make the
package compute wrong results or break users:

- Bugs and logic errors: wrong edge-case behavior, or crashes on valid
  input.
- Incorrect math or statistics: wrong sign, missing normalization or
  Jacobian factor, mishandled log/exp (e.g. overflow where a
  logsumexp/`log1p` formulation is needed), biased sampling.
- Violations of the array dimension convention: layer dimensions first,
  batch dimension last; weights `w` shaped
  `(size(visible)..., size(hidden)...)`. Code that only works for
  1-dimensional layers, a single batch, or non-Potts layers — while
  claiming to be general — is a bug, not a nit.
- Code that breaks generic array-backend support (CUDA via `ext/`, tested
  with JLArrays and `allowscalar(false)`): scalar indexing into device
  arrays, constructing `Array`/`Vector` where `similar` or broadcasting is
  needed, or dispatch signatures that exclude non-`Array` backends.
- Layer parameter storage violations: parameters must live in the single
  `.par` array with named accessors as views; a layer type that stores
  parameters elsewhere breaks HDF5 persistence and gauge transformations.
- Breaking changes to exported API or behavior not reflected in tests and
  in the `## Unreleased` section of CHANGELOG.md.
- Security issues in changes to CI workflows.

Style, naming, refactoring suggestions, and docstring wording are Nit at
most.

## Verification bar

A claim that math is wrong needs a short derivation or a citation of the
correct form elsewhere in this codebase (`file:line`) — not an inference
from function names. A claim that code breaks on GPU arrays or on
higher-dimensional layers needs the concrete failing call shape, not a
suspicion.

## Cap the nits

Report at most five Nits per review. If you found more, say "plus N similar
items" in the summary instead of posting them inline. If everything you
found is a Nit, lead the summary with "No blocking issues."

## Do not report

- Anything CI already enforces: the test suite (`ci.yml`) and the
  agent-docs linter (`.github/scripts/lint_agent_docs.py`).
- Formatting and code style preferences.
- Generated files: `Manifest.toml`, built documentation output, and
  anything under `notebooks/`.
- Missing CHANGELOG.md entries for CI, workflow, or repo-tooling changes —
  the changelog records only user-facing package changes.

## Always check

- New or changed layer types implement the full interface: `energy`,
  `cgfs`, `sample_from_inputs`, `mean_from_inputs`, `var_from_inputs`,
  `mode_from_inputs`.
- New tests do not require a physical GPU (GitHub CI has none); GPU
  compatibility belongs in `test/jlarrays.jl` via JLArrays.
- Changes to exported functions keep docstrings and tests in sync with the
  new behavior.
- The version in Project.toml keeps its `-DEV` suffix outside of release
  PRs.

## Agent-instruction files

When the diff touches CLAUDE.md, AGENTS.md, REVIEW.md, or anything under
`.agents/` or `.claude/`, also review those files for: contradictions with
each other or with the actual repository (spot-check commands, paths, and
factual claims against the code); substantial redundancy that can drift
apart; context bloat (content that does not earn its place); and skill
frontmatter descriptions that fail to say what the skill does and when to
use it. Consult the best-practice guides with WebFetch if helpful:
https://code.claude.com/docs/en/best-practices and
https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
The deterministic linter (`.github/scripts/lint_agent_docs.py`) already
enforces sizes, frontmatter constraints, and path existence — focus on
semantics.

## Re-reviews

After the first review of a PR, suppress new Nits and post Important
findings only, so follow-up pushes converge instead of accumulating style
rounds.
