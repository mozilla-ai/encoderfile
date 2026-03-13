# CVE Semantic Search — Encoderfile + Qdrant POC

A fully local, privacy-first vulnerability search system demonstrating
how [Encoderfile](https://github.com/mozilla-ai/encoderfile) and
[Qdrant](https://qdrant.tech/) work together.

**Encoderfile** generates embeddings locally.
**Qdrant** stores and searches vectors — self-hosted, no data leaves the network.

## The Use Case

Internal security teams need to search across their vulnerability reports, pen test findings, and bug bounty submissions using natural language. Queries like "authentication bypass in our VPN infrastructure" should return semantically relevant results — not just keyword matches.

The data is too sensitive for cloud embedding APIs. Encoderfile + Qdrant
keeps everything local.

### Key Characteristics

- No Python ML Runtime: Encoderfile runs as a single, compiled binary.

- Air-gapped by Default: Zero external network calls required at inference time.

- Auditable: The embedding binary is reproducible and hashable for strict compliance environments.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

To include the FastEmbed fallback:

```bash
uv sync --extra fastembed
```

### 2. Download and Start Encoderfile

Download the `all-MiniLM-L6-v2` binary for your specific OS and architecture (e.g., macOS ARM, Linux x86) from the [Mozilla AI Hugging Face repository](https://huggingface.co/mozilla-ai/encoderfile/tree/main/sentence-transformers/all-MiniLM-L6-v2).

Once downloaded, make it executable and start the server:

```bash
chmod +x all-MiniLM-L6-v2.encoderfile
./all-MiniLM-L6-v2.encoderfile serve --http-port 8080
```

### 3. Run the demo

```bash
# Run with default example query
python main.py

# Run a specific query
python main.py -q "SQL injection in file transfer software"

# Interactive search mode
python main.py --interactive
```

The demo auto-detects whether Encoderfile is running. If not, it falls back
to FastEmbed (Qdrant's own embedding library, same model) so you can see
the integration working regardless.

## What It Does

1. Embeds 30 real CVE descriptions (from NVD's public database)
2. Stores the vectors + metadata in Qdrant
3. Runs a query and shows ranked results

Try queries like:
- `authentication bypass in VPN products`
- `SQL injection in file transfer software`
- `supply chain compromise through software updates`
- `zero-day exploited by nation-state actors`

## File Structure

```
cve-search-poc/
├── main.py          # Ingest → search, with optional --interactive mode
├── embedders.py     # Embedding providers (Encoderfile primary, FastEmbed fallback)
├── cve_data.json    # Sample CVE data (30 real CVEs from NVD)
├── pyproject.toml   # uv project definition and dependencies
└── README.md
```

## Extending This

**More data**: Replace `cve_data.json` with a script that pulls from
[NVD's API](https://nvd.nist.gov/developers/vulnerabilities) or
ingests your internal vulnerability reports.

**Full RAG**: Add [Llamafile](https://github.com/Mozilla-Ocho/llamafile)
as the generation layer. Retrieve relevant CVEs from Qdrant, pass them
as context to a local LLM, get natural language summaries.

**Production Qdrant**: Swap `QdrantClient(":memory:")` for
`QdrantClient("localhost", port=6333)` pointed at a Docker or
bare-metal Qdrant instance with persistence.

