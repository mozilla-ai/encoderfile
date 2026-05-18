# GitBook Documentation Branch

The `gitbook-docs` branch contains **generated** GitBook-compatible documentation,
automatically updated by GitHub Actions on every push to `main`.

**Do not edit this branch manually** — all changes will be overwritten.

## How it works

1. `scripts/convert_to_gitbook.py` copies `docs/` into `site/`, converts
   MkDocs-specific syntax (snippets, admonitions, tabs, footnotes) to
   GitBook-compatible Markdown, and writes `SUMMARY.md`
2. The contents of `site/` are pushed to this branch
3. GitBook syncs from this branch
