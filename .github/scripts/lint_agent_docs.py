#!/usr/bin/env python3
"""Lint agent-instruction files (CLAUDE.md, AGENTS.md, agent skills).

Deterministic checks that keep these files honest and slim:

- size budgets, so the context these files consume stays small
- SKILL.md frontmatter rules from the Agent Skills spec
  (https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- repo paths mentioned in the docs must exist (staleness check)
- ``--project=<dir>`` flags must point at real Julia projects
- long lines duplicated verbatim across files are reported (redundancy signal)

Semantic checks (contradictions, redundancy in meaning, guideline adherence)
are out of scope here — review those by hand or with an agent when the docs
change substantially.

Errors fail CI (exit 1); warnings are annotated but do not fail.
Runs with the Python 3 standard library only (PyYAML is used when available).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LIMITS = {
    # CLAUDE.md / AGENTS.md are loaded into context on every session.
    "memory_lines": 250,
    "memory_bytes": 16_000,
    # Agent Skills spec: keep SKILL.md under 500 lines, push detail to references.
    "skill_lines": 500,
    "skill_name_len": 64,
    "skill_desc_len": 1024,
    "reference_lines": 1000,
    # Below this length a description cannot say what the skill does *and* when to use it.
    "skill_desc_min": 40,
    # Lines shorter than this are too generic for the duplicate check.
    "dup_min_len": 45,
}

MEMORY_FILES = ["CLAUDE.md", "AGENTS.md", ".claude/CLAUDE.md"]
SKILL_GLOBS = [".agents/skills/*/SKILL.md", ".claude/skills/*/SKILL.md"]

# Tokens starting with these are treated as repo paths that must exist.
DIR_PREFIXES = (
    "src/", "test/", "docs/", "ext/", "repl/", "notebooks/", "example/",
    "references/", "agents/", ".claude/", ".agents/", ".github/",
)
# Bare filenames treated as repo-root paths (committed files only —
# Manifest.toml is intentionally absent because it is gitignored).
TOP_LEVEL_FILES = {
    "CLAUDE.md", "AGENTS.md", "README.md", "CHANGELOG.md", "Project.toml",
    "LICENSE.md", "CITATION.cff", "codecov.yml", "SKILL.md",
}
# Tokens containing these (case-insensitive) are placeholders, not real paths.
PLACEHOLDER_MARKERS = ("<", ">", "$", "path/to", "your")

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
KNOWN_SKILL_KEYS = {"name", "description", "version", "license", "allowed-tools", "metadata"}

TOKEN_SPLIT = re.compile(r"""[\s'"()\[\]{},;`|]+""")
PROJECT_FLAG_RE = re.compile(r"--project=([^\s'\"`,)\]]+)")

errors: list[tuple[Path, int, str]] = []
warnings: list[tuple[Path, int, str]] = []


def error(path: Path, line: int, msg: str) -> None:
    errors.append((path, line, msg))


def warn(path: Path, line: int, msg: str) -> None:
    warnings.append((path, line, msg))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def parse_frontmatter(text: str):
    """Return (frontmatter dict or None, error message or None)."""
    m = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    if not m:
        return None, "missing YAML frontmatter (--- ... ---) at the top of the file"
    block = m.group(1)
    try:
        import yaml  # available on GitHub runners; optional locally
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError as exc:
            return None, f"frontmatter is not valid YAML: {' '.join(str(exc).split())}"
        if not isinstance(data, dict):
            return None, "frontmatter is not a YAML mapping"
        return data, None
    # Minimal fallback: top-level `key: value` and `key: >`/`key: |` folded blocks.
    data: dict[str, str] = {}
    key = None
    folded: list[str] = []
    for line in block.splitlines():
        if line[:1] not in (" ", "\t") and ":" in line:
            if key is not None:
                data[key] = " ".join(folded).strip()
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val in (">", "|", ">-", "|-"):
                folded = []
            else:
                data[key] = val
                key = None
        elif key is not None and line.strip():
            folded.append(line.strip())
    if key is not None:
        data[key] = " ".join(folded).strip()
    return data, None


def code_tokens(path: Path):
    """Yield (line_number, token) for tokens inside inline code spans and fenced blocks."""
    in_fence = False
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        chunks = [line] if in_fence else re.findall(r"`([^`]+)`", line)
        for chunk in chunks:
            for tok in TOKEN_SPLIT.split(chunk):
                tok = tok.rstrip(".,:;")  # rstrip only: `.claude/...` keeps its leading dot
                if tok:
                    yield lineno, tok


def is_placeholder(tok: str) -> bool:
    low = tok.lower()
    return any(marker in low for marker in PLACEHOLDER_MARKERS)


def path_exists(tok: str, doc: Path) -> bool:
    tok = tok.rstrip("/")
    bases = [ROOT]
    if doc.parent != ROOT:
        bases.insert(0, doc.parent)  # skill docs may use paths relative to the skill dir
    for base in bases:
        if "*" in tok:
            try:
                if any(base.glob(tok)):
                    return True
            except (ValueError, NotImplementedError):
                pass
        elif (base / tok).exists():
            return True
    return False


def check_paths(doc: Path) -> None:
    seen: set[str] = set()
    for lineno, tok in code_tokens(doc):
        tok = re.sub(r":\d+(-\d+)?$", "", tok)  # `src/rbm.jl:42` cites a line, not a path
        if tok in seen or is_placeholder(tok):
            continue
        if not (tok in TOP_LEVEL_FILES or tok.startswith(DIR_PREFIXES)):
            continue
        seen.add(tok)
        if not path_exists(tok, doc):
            error(doc, lineno, f"references `{tok}`, which does not exist in the repository")


def check_project_flags(doc: Path) -> None:
    text = doc.read_text()
    for lineno, line in enumerate(text.splitlines(), start=1):
        for proj in PROJECT_FLAG_RE.findall(line):
            if proj in (".", "@.") or is_placeholder(proj):
                continue
            if not (ROOT / proj / "Project.toml").is_file():
                error(doc, lineno, f"`--project={proj}` but {proj}/Project.toml does not exist")


def check_size(doc: Path, max_lines: int, max_bytes: int | None = None) -> None:
    n_lines = len(doc.read_text().splitlines())
    n_bytes = doc.stat().st_size
    if n_lines > max_lines:
        error(doc, 1, f"{n_lines} lines exceeds the {max_lines}-line budget; move detail into a skill reference or delete it")
    elif n_lines > 0.8 * max_lines:
        warn(doc, 1, f"{n_lines} lines is close to the {max_lines}-line budget")
    if max_bytes is not None and n_bytes > max_bytes:
        error(doc, 1, f"{n_bytes} bytes exceeds the {max_bytes}-byte budget")


def check_skill(skill_md: Path, seen_names: dict[str, Path]) -> None:
    text = skill_md.read_text()
    fm, err = parse_frontmatter(text)
    if err:
        error(skill_md, 1, err)
        return

    name = fm.get("name")
    if not isinstance(name, str) or not name:
        error(skill_md, 1, "frontmatter is missing a `name`")
    else:
        if len(name) > LIMITS["skill_name_len"]:
            error(skill_md, 1, f"skill name is {len(name)} chars (max {LIMITS['skill_name_len']})")
        if not SKILL_NAME_RE.match(name):
            error(skill_md, 1, f"skill name `{name}` must be lowercase letters, digits, and hyphens")
        if name != skill_md.parent.name:
            error(skill_md, 1, f"skill name `{name}` does not match its directory `{skill_md.parent.name}`")
        if name in seen_names:
            error(skill_md, 1, f"skill name `{name}` is also used by {rel(seen_names[name])}")
        else:
            seen_names[name] = skill_md

    desc = fm.get("description")
    if not isinstance(desc, str) or not desc.strip():
        error(skill_md, 1, "frontmatter is missing a `description`")
    else:
        desc = " ".join(desc.split())
        if len(desc) > LIMITS["skill_desc_len"]:
            error(skill_md, 1, f"description is {len(desc)} chars (max {LIMITS['skill_desc_len']})")
        if len(desc) < LIMITS["skill_desc_min"]:
            error(skill_md, 1, "description is too short to say what the skill does and when to use it")

    for key in fm:
        if key not in KNOWN_SKILL_KEYS:
            warn(skill_md, 1, f"unexpected frontmatter key `{key}`")

    check_size(skill_md, LIMITS["skill_lines"])

    for ref in skill_md.parent.rglob("*.md"):
        if ref != skill_md:
            check_size(ref, LIMITS["reference_lines"])


def check_duplicates(docs: list[Path]) -> None:
    lines_by_content: dict[str, list[tuple[Path, int]]] = {}
    for doc in docs:
        for lineno, line in enumerate(doc.read_text().splitlines(), start=1):
            norm = " ".join(line.split())
            if len(norm) < LIMITS["dup_min_len"] or norm.startswith(("#", "```", "-->", "<!--")):
                continue
            lines_by_content.setdefault(norm, []).append((doc, lineno))
    for norm, hits in lines_by_content.items():
        files = {doc for doc, _ in hits}
        if len(files) > 1:
            doc, lineno = hits[0]
            others = ", ".join(sorted(rel(d) for d in files if d != doc))
            warn(doc, lineno, f"line duplicated verbatim in {others}: `{norm[:80]}`")


def main() -> int:
    memory_docs = [ROOT / f for f in MEMORY_FILES if (ROOT / f).is_file()]
    skill_docs = sorted(p for g in SKILL_GLOBS for p in ROOT.glob(g))
    reference_docs = sorted(
        ref for s in skill_docs for ref in s.parent.rglob("*.md") if ref != s
    )

    if not memory_docs:
        warn(ROOT / "CLAUDE.md", 1, "no CLAUDE.md or AGENTS.md found")

    for doc in memory_docs:
        check_size(doc, LIMITS["memory_lines"], LIMITS["memory_bytes"])
    seen_names: dict[str, Path] = {}
    for skill in skill_docs:
        check_skill(skill, seen_names)

    all_docs = memory_docs + skill_docs + reference_docs
    for doc in all_docs:
        check_paths(doc)
        check_project_flags(doc)
    check_duplicates(all_docs)

    print(f"Checked {len(all_docs)} agent doc(s): " + ", ".join(rel(d) for d in all_docs))
    for path, lineno, msg in warnings:
        print(f"::warning file={rel(path)},line={lineno}::{msg}")
        print(f"WARNING {rel(path)}:{lineno}: {msg}")
    for path, lineno, msg in errors:
        print(f"::error file={rel(path)},line={lineno}::{msg}")
        print(f"ERROR {rel(path)}:{lineno}: {msg}")
    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"\nOK ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
