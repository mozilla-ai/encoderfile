# Local RAG with Encoderfile + Llamafile


A fully local RAG (Retrieval-Augmented Generation) system. Give it any text file, ask it questions. Nothing leaves your machine.

## Quickstart

### 1. Download the binaries

```bash
# Llama 3.2 3B Instruct (~2.5GB)
wget -O llama3.2-3b.llamafile \
  https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/resolve/main/Llama-3.2-3B-Instruct.Q6_K.llamafile
chmod +x llama3.2-3b.llamafile

# all-MiniLM-L6-v2 encoder (~80MB) — pick the file for your architecture:
# macOS (Apple Silicon):  all-MiniLM-L6-v2-aarch64-apple-darwin.encoderfile
# macOS (Intel):          all-MiniLM-L6-v2-x86_64-apple-darwin.encoderfile
# Linux (x86_64):         all-MiniLM-L6-v2-x86_64-unknown-linux-gnu.encoderfile
# Windows (x86_64):       all-MiniLM-L6-v2-x86_64-pc-windows-msvc.encoderfile
# Browse all builds:      https://huggingface.co/mozilla-ai/encoderfile/tree/main/sentence-transformers/all-MiniLM-L6-v2
wget -O minilm.encoderfile <PASTE_URL_FOR_YOUR_ARCHITECTURE>
chmod +x minilm.encoderfile
```

> **Windows:** Skip `chmod`. Instead, rename each file to add a `.exe` extension (e.g. `llama3.2-3b.llamafile.exe`) and run it directly.

### 2. Start the servers

```bash
# Terminal 1
./minilm.encoderfile serve --http-port 8080

# Terminal 2
./llama3.2-3b.llamafile --server --port 8081 --nobrowser
```

### 3. Run

```bash
uv sync
uv run local_rag.py
```

You'll get an interactive prompt where you can ask questions about the included sample datasets:

- `weird_laws.txt` — a collection of unusual US laws, good for testing separator-mode chunking
- `genius_act.txt` — the GENIUS Act (S. 919, 119th Congress), a stablecoin regulation bill, good for testing window-mode chunking on legislative prose

```bash
# Ask about weird laws (default)
uv run local_rag.py

# Ask about the GENIUS Act
uv run local_rag.py --file genius_act.txt --chunk-mode window --chunk-size 800
```

## Use Your Own Data

Point it at any text file:

```bash
# A congressional bill
uv run local_rag.py --file bill.txt --chunk-mode window --chunk-size 800

# A structured dataset (entries separated by blank lines)
uv run local_rag.py --file my_notes.txt --chunk-mode separator

# Customize everything
uv run local_rag.py \
  --file contracts.txt \
  --chunk-mode window \
  --chunk-size 1000 \
  --chunk-overlap 100 \
  --top-k 3 \
  --system-prompt "You are a legal assistant. Answer using only the provided context."
```

### Chunking modes

- **`separator`** (default) — Splits on blank lines. Best for structured data where each entry is self-contained.
- **`window`** — Splits into overlapping character windows. Best for raw prose like legislation, articles, or books.

## How It Works

1. **Embed** — Each text chunk is converted into a 384-dimensional vector by encoderfile (all-MiniLM-L6-v2).
2. **Retrieve** — Your question is embedded the same way, and the closest chunks are found via cosine similarity.
3. **Generate** — Retrieved chunks are passed as context to Llama 3.2 3B (via llamafile), which generates an answer.

No vector database — numpy handles similarity search in memory. This is fast for datasets up to a few thousand chunks.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | `weird_laws.txt` | Path to your text file |
| `--chunk-mode` | `separator` | `separator` or `window` |
| `--chunk-separator` | `\n\n` | String to split on (separator mode) |
| `--chunk-size` | `500` | Characters per chunk (window mode) |
| `--chunk-overlap` | `50` | Overlap between chunks (window mode) |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--encoderfile-url` | `http://localhost:8080` | Encoderfile server URL |
| `--llamafile-url` | `http://localhost:8081/v1` | Llamafile server URL |
| `--system-prompt` | Generic helpful assistant | System prompt for the LLM |

## Going Further

**Better answers:** Use a larger llamafile — [Mistral 7B](https://huggingface.co/mozilla-ai/Mistral-7B-Instruct-v0.2-llamafile) or [Llama 3.1 8B](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile).

**Better retrieval:** Use a larger encoder via encoderfile, like `all-mpnet-base-v2` or `BGE-small`.

**Swap models freely:** Use [any-llm](https://github.com/mozilla-ai/any-llm) to compare llamafile, OpenAI, Gemini, and Claude with one config change.

**Add orchestration & evals:** Wrap this in [any-agent](https://mozilla-ai.github.io/any-agent/) for tracing and evaluation.

---

*Built with [encoderfile](https://github.com/mozilla-ai/encoderfile) and [llamafile](https://github.com/mozilla-ai/llamafile) from [Mozilla AI](https://www.mozilla.ai/).*