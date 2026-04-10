"""Build the GitBook site output into site/.

Copies docs/ into site/, converts MkDocs-specific syntax to GitBook-compatible
Markdown (snippets, admonitions, tabs, footnotes), and writes SUMMARY.md.
The contents of site/ are pushed to the gitbook-docs branch by CI.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_SRC = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "site"

# Maps MkDocs admonition types to GitBook hint styles
ADMONITION_STYLE: dict[str, str] = {
    "note": "info",
    "info": "info",
    "tip": "success",
    "success": "success",
    "warning": "warning",
    "caution": "warning",
    "danger": "danger",
    "error": "danger",
    "failure": "danger",
    "question": "info",
    "abstract": "info",
    "example": "info",
    "quote": "info",
    "bug": "warning",
}

SUMMARY = """\
# Table of Contents

* [Introduction](index.md)
* [Getting Started](getting-started.md)

## Building Encoderfiles

* [Building with Docker](building_encoderfiles/docker.md)

## Python Library

* [Building with Python](python/building-with-python.md)
* [API Reference](python/api-reference.md)

## Cookbooks

* [Token Classification (NER)](cookbooks/token-classification-ner.md)
* [MCP Integration](cookbooks/mcp-integration.md)
* [Matryoshka Embeddings](cookbooks/matryoshka-embeddings.md)
* [Local RAG](cookbooks/local-rag.md)
* [CVE Semantic Search](cookbooks/qdrant-cve-search.md)

## Transforms

* [Transforms](transforms/index.md)
* [Reference](transforms/reference.md)

## Reference

* [Encoderfile File Format](reference/file_format.md)
* [CLI Reference](reference/cli.md)
* [API Reference](reference/api-reference.md)
* [Building Guide](reference/building.md)

## Community

* [Contributing](CONTRIBUTING.md)
* [Code of Conduct](CODE_OF_CONDUCT.md)
"""


def inline_snippets(text: str) -> str:
    """Replace --8<-- "path" markers with the contents of the referenced file."""

    def replace_snippet(match: re.Match) -> str:
        snippet_path = REPO_ROOT / match.group(1)
        if snippet_path.exists():
            return snippet_path.read_text(encoding="utf-8").rstrip()
        print(f"  WARNING: snippet not found: {snippet_path}")
        return match.group(0)

    return re.sub(r'--8<--\s+"([^"]+)"', replace_snippet, text)


def convert_tabs(text: str) -> str:
    """Convert MkDocs === "Tab" blocks to GitBook tab syntax."""
    lines = text.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if re.match(r'^===\s+"', line):
            # Collect all consecutive tab blocks (blank lines between them are ok)
            tabs: list[tuple[str, list[str]]] = []
            while i < len(lines):
                m = re.match(r'^===\s+"([^"]+)"', lines[i])
                if m:
                    tab_title = m.group(1)
                    i += 1
                    tab_lines: list[str] = []
                    while i < len(lines):
                        bl = lines[i]
                        if bl == "" or bl.startswith("    "):
                            tab_lines.append(bl)
                            i += 1
                        else:
                            break
                    # Strip trailing blank lines inside tab
                    while tab_lines and tab_lines[-1] == "":
                        tab_lines.pop()
                    tabs.append((tab_title, tab_lines))
                elif lines[i] == "":
                    # Blank line between tab blocks — skip, but stop if next
                    # non-blank line is not another tab
                    j = i + 1
                    while j < len(lines) and lines[j] == "":
                        j += 1
                    if j < len(lines) and re.match(r'^===\s+"', lines[j]):
                        i += 1  # skip blank line and keep looking for tabs
                    else:
                        break
                else:
                    break

            result.append("{% tabs %}")
            for tab_title, tab_lines in tabs:
                result.append(f'{{% tab title="{tab_title}" %}}')
                for bl in tab_lines:
                    result.append(bl[4:] if bl.startswith("    ") else bl)
                result.append("{% endtab %}")
            result.append("{% endtabs %}")
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def convert_admonitions(text: str) -> str:
    """Convert MkDocs !!! and ??? admonition blocks to GitBook hint blocks."""
    lines = text.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(!{3}|\?{3})\s+(\w+)(?:\s+"([^"]*)")?', line)
        if m:
            admon_type = m.group(2).lower()
            style = ADMONITION_STYLE.get(admon_type, "info")
            title = m.group(3)

            i += 1
            block_lines: list[str] = []
            while i < len(lines):
                bl = lines[i]
                if bl == "" or bl.startswith("    "):
                    block_lines.append(bl)
                    i += 1
                else:
                    break

            # Strip trailing blank lines
            while block_lines and block_lines[-1] == "":
                block_lines.pop()

            result.append(f'{{% hint style="{style}" %}}')
            if title:
                result.append(f"**{title}**")
                result.append("")
            for bl in block_lines:
                result.append(bl[4:] if bl.startswith("    ") else bl)
            result.append("{% endhint %}")
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def convert_footnotes(text: str) -> str:
    """Convert Markdown footnotes to inline GitBook hint blocks."""
    footnote_defs: dict[str, str] = {}

    def collect_def(m: re.Match) -> str:
        footnote_defs[m.group(1)] = m.group(2).strip()
        return ""

    text = re.sub(r"^\[\^(\w+)\]:\s*(.+)$", collect_def, text, flags=re.MULTILINE)
    text = re.sub(r"\[\^(\w+)\]", "", text)

    if footnote_defs:
        notes = "\n\n".join(
            f'{{% hint style="info" %}}\n{content}\n{{% endhint %}}'
            for content in footnote_defs.values()
        )
        text = text.rstrip() + "\n\n" + notes + "\n"

    return text


def convert_file(text: str) -> str:
    """Apply all MkDocs → GitBook conversions to a markdown file."""
    text = inline_snippets(text)
    text = convert_tabs(text)
    text = convert_admonitions(text)
    text = convert_footnotes(text)
    return text


def main() -> None:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    shutil.copytree(DOCS_SRC, SITE_DIR)

    md_files = list(SITE_DIR.rglob("*.md"))
    for md_path in md_files:
        original = md_path.read_text(encoding="utf-8")
        converted = convert_file(original)
        if converted != original:
            md_path.write_text(converted, encoding="utf-8")
            print(f"  converted: {md_path.relative_to(SITE_DIR)}")

    (SITE_DIR / "SUMMARY.md").write_text(SUMMARY, encoding="utf-8")
    print(f"\nDone — {len(md_files)} files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
