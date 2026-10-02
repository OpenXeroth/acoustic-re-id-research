"""Fail CI on broken relative Markdown links or undocumented top-level docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
REMOTE_SCHEMES = ("http://", "https://", "mailto:")


def markdown_files() -> list[Path]:
    return sorted(
        [
            ROOT / "README.md",
            ROOT / "AGENTS.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "SECURITY.md",
            *DOCS.rglob("*.md"),
        ]
    )


def validate_link(source: Path, target: str) -> str | None:
    target = target.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("#", *REMOTE_SCHEMES)):
        return None
    path_text = unquote(target.split("#", 1)[0])
    if path_text.startswith(("/", "file:")):
        return f"absolute/local-only link: {target}"
    resolved = (source.parent / path_text).resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        return f"link escapes the repository: {target}"
    if not resolved.exists():
        return f"missing target: {target}"
    return None


def main() -> int:
    failures: list[str] = []
    files = markdown_files()
    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            failure = validate_link(source, match.group(1))
            if failure:
                failures.append(f"{source.relative_to(ROOT)}: {failure}")

    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for document in sorted(DOCS.glob("*.md")):
        if document.name != "index.md" and document.name not in index:
            failures.append(f"docs/index.md: top-level document not indexed: {document.name}")

    experiments = DOCS / "experiments"
    listing = (experiments / "README.md").read_text(encoding="utf-8")
    for document in sorted(experiments.glob("*.md")):
        if document.name not in {"README.md", "TEMPLATE.md"} and document.name not in listing:
            failures.append(f"docs/experiments/README.md: experiment not listed: {document.name}")

    if failures:
        print("Documentation integrity failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Documentation integrity passed for {len(files)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
