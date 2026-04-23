"""Validate checked-in docs before publishing.

The main goal is to catch broken internal links locally before docs changes make
it to GitBook. External links are intentionally skipped so the checker stays
fast and works offline.

Usage:
    uv run python scripts/check_docs.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
GITBOOK_CONFIG = REPO_ROOT / ".gitbook.yaml"
LINK_RE = re.compile(r"!?[[][^]]*[]][(]([^)\n]+)[)]")
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CODE_FENCE_RE = re.compile(r"^```")
SKIPPED_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")


def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks so code samples are not linted as page links."""
    output: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            output.append("")
            continue
        output.append("" if in_fence else line)

    return "\n".join(output)


def slugify_heading(raw_heading: str) -> str:
    """Approximate the anchor slugs used by common Markdown site generators."""
    heading = re.sub(r"`([^`]*)`", r"\1", raw_heading.strip().lower())
    heading = re.sub(r"[^\w\s-]", "", heading)
    heading = re.sub(r"\s+", "-", heading)
    heading = re.sub(r"-{2,}", "-", heading)
    return heading.strip("-")


def extract_anchors(path: Path) -> set[str]:
    """Collect heading anchors from a Markdown document."""
    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADER_RE.match(line)
        if match:
            anchors.add(slugify_heading(match.group(2)))
    return anchors


def split_target(raw_target: str) -> tuple[str, str]:
    """Split a Markdown link target into path and optional anchor."""
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]

    if " " in target and not target.startswith("#"):
        target = target.split(" ", 1)[0]

    if "#" in target:
        path_part, anchor = target.split("#", 1)
        return path_part, anchor

    return target, ""


def resolve_target(source_path: Path, target_path: str) -> Path:
    """Resolve a relative docs link target against the source document."""
    base = source_path.parent
    resolved = (base / target_path).resolve()
    if resolved.exists():
        return resolved

    if resolved.suffix == "":
        markdown_candidate = resolved.with_suffix(".md")
        if markdown_candidate.exists():
            return markdown_candidate

        index_candidate = resolved / "index.md"
        if index_candidate.exists():
            return index_candidate

    return resolved


def validate_gitbook_config(errors: list[str]) -> None:
    """Ensure the configured readme and summary exist in docs/."""
    config_text = GITBOOK_CONFIG.read_text(encoding="utf-8")
    for key in ("readme", "summary"):
        match = re.search(rf"^\s*{key}:\s*(\S+)\s*$", config_text, re.MULTILINE)
        if not match:
            errors.append(f".gitbook.yaml is missing `{key}`")
            continue

        path = DOCS_DIR / match.group(1)
        if not path.exists():
            errors.append(
                f".gitbook.yaml points `{key}` to missing file: {path.relative_to(REPO_ROOT)}"
            )


def main() -> int:
    """Validate docs links and anchors. Returns a process exit code."""
    errors: list[str] = []
    anchors_by_file = {
        path.resolve(): extract_anchors(path) for path in DOCS_DIR.rglob("*.md")
    }

    validate_gitbook_config(errors)

    for source_path in sorted(DOCS_DIR.rglob("*.md")):
        text = strip_code_blocks(source_path.read_text(encoding="utf-8"))

        for match in LINK_RE.finditer(text):
            raw_target = match.group(1)
            if raw_target.startswith(SKIPPED_PREFIXES):
                continue

            target_path, anchor = split_target(raw_target)

            if target_path == "":
                target_file = source_path.resolve()
            else:
                target_file = resolve_target(source_path, target_path)
                if not target_file.exists():
                    errors.append(
                        f"{source_path.relative_to(REPO_ROOT)} -> missing target `{target_path}`"
                    )
                    continue

            if anchor:
                target_anchors = anchors_by_file.get(target_file.resolve())
                if target_anchors is None:
                    continue

                if slugify_heading(anchor) not in target_anchors:
                    errors.append(
                        f"{source_path.relative_to(REPO_ROOT)} -> missing anchor `#{anchor}` in "
                        f"{target_file.relative_to(REPO_ROOT)}"
                    )

    if errors:
        print("Documentation checks failed:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Documentation checks passed for {len(anchors_by_file)} markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
