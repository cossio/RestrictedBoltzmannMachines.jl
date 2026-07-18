# Claude pull-request review

Follow `REVIEW.md`. Inspect the full pull-request diff and enough surrounding
code to judge each change in context.

Post only actionable findings as inline comments using
`mcp__github_inline_comment__create_inline_comment` with `confirmed: true`.
Do not report style preferences or duplicate existing comments. If there are
no actionable findings, post no inline comments. Summarize what you checked in
the final response.

Never modify code, push commits, merge the pull request, or enable auto-merge.
