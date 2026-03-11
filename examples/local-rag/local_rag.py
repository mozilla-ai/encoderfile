"""
Local RAG with Encoderfile + Llamafile.

Zero cloud. Zero API keys. Two binaries and this script.

Usage:
    python local_rag.py                                                        # uses weird_laws.txt
    python local_rag.py --file genius_act.txt --chunk-mode window              # GENIUS Act (stablecoin bill)
    python local_rag.py --file my_document.txt                                 # your own file
    python local_rag.py --file bill.txt --chunk-mode window --chunk-size 800   # 800 chars (~160 tokens)
"""

import argparse
import re
import numpy as np
import requests

# ── Text Cleaning ──


def clean_text(text):
    """Strip common formatting cruft from legislative and web-sourced text."""
    # Remove XML/HTML-style tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    for entity, char in [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&mdash;", "—"),
        ("&ndash;", "–"),
        ("&sect;", "§"),
    ]:
        text = text.replace(entity, char)
    # Remove any remaining numeric HTML entities
    text = re.sub(r"&#\d+;", "", text)
    # Collapse lines of underscores, dashes, equals signs (visual separators)
    text = re.sub(r"[_\-=]{5,}", "", text)
    # Collapse runs of whitespace on a single line
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse multiple blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Embed ──


def embed(texts, encoderfile_url, batch_size=32):
    """Embed texts via encoderfile, mean-pooling token embeddings.
    Batches requests to avoid overwhelming the server with large payloads."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(f"{encoderfile_url}/predict", json={"inputs": batch})
        resp.raise_for_status()
        for result in resp.json()["results"]:
            token_embs = [e["embedding"] for e in result["embeddings"]]
            all_embeddings.append(np.mean(token_embs, axis=0))
    return np.array(all_embeddings)


# ── Chunk ──

MIN_CHUNK_LENGTH = 20  # Skip chunks shorter than this (formatting artifacts)


def chunk_by_separator(text, separator="\n\n"):
    """Split text on a separator. Good for structured entries."""
    return [
        c.strip()
        for c in text.split(separator)
        if c.strip() and len(c.strip()) >= MIN_CHUNK_LENGTH
    ]


def chunk_by_window(text, size=500, overlap=50):
    """Split text into overlapping windows. Good for raw prose.
    size and overlap are in characters (~4-5 chars per token)."""
    step = size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + size].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
    return chunks


# ── Retrieve ──


def retrieve(query, chunks, chunk_embeddings, encoderfile_url, top_k=5):
    """Find the top-k most relevant chunks for a query."""
    query_emb = embed([query], encoderfile_url)

    # Cosine similarity with safe norms
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_emb, keepdims=True)

    # Mask out chunks with zero/bad norms instead of dividing
    valid = (chunk_norms.squeeze() > 1e-8) & np.isfinite(chunk_norms.squeeze())
    scores = np.full(len(chunks), -1.0)  # default: low score
    if valid.any() and query_norm > 1e-8:
        normed_chunks = chunk_embeddings[valid] / chunk_norms[valid]
        normed_query = query_emb / query_norm
        scores[valid] = (normed_chunks @ normed_query.T).squeeze()

    # Replace any remaining NaN/inf with -1
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    top = np.argsort(scores)[-top_k:][::-1]
    return [(chunks[i], float(scores[i])) for i in top]


# ── Generate ──


def ask(
    question,
    chunks,
    chunk_embeddings,
    encoderfile_url,
    llamafile_url,
    system_prompt,
    top_k=5,
):
    """Retrieve relevant chunks, then generate an answer."""
    results = retrieve(question, chunks, chunk_embeddings, encoderfile_url, top_k)
    context = "\n\n".join(chunk for chunk, _ in results)

    resp = requests.post(
        f"{llamafile_url}/chat/completions",
        timeout=120,
        json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            "temperature": 0.1,
        },
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    # Strip LLM control tokens that sometimes leak into output
    answer = re.sub(r"<\|[^|]+\|>", "", answer).strip()
    return answer


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG with encoderfile + llamafile"
    )
    parser.add_argument("--file", default="weird_laws.txt", help="Path to text file")
    parser.add_argument(
        "--chunk-mode",
        choices=["separator", "window"],
        default="separator",
        help="'separator' splits on blank lines, 'window' uses character windows",
    )
    parser.add_argument(
        "--chunk-separator",
        default="\n\n",
        help="Separator string (default: blank line)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Window size in characters (~100-120 tokens). Not token count.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks in characters",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of chunks to retrieve"
    )
    parser.add_argument("--encoderfile-url", default="http://localhost:8080")
    parser.add_argument("--llamafile-url", default="http://localhost:8081/v1")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful assistant that answers questions using ONLY the provided context. "
            "If the context doesn't contain enough information, say so."
        ),
    )
    args = parser.parse_args()

    # Load & clean
    with open(args.file) as f:
        text = clean_text(f.read())

    # Chunk
    if args.chunk_mode == "separator":
        chunks = chunk_by_separator(text, args.chunk_separator)
    else:
        chunks = chunk_by_window(text, args.chunk_size, args.chunk_overlap)
    print(f"Loaded {len(chunks)} chunks from {args.file}")

    # Index
    print("Embedding chunks...")
    try:
        chunk_embeddings = embed(chunks, args.encoderfile_url)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to encoderfile at {args.encoderfile_url}.")
        print("Make sure your encoderfile server is running.")
        return
    n_bad = np.sum(
        ~np.isfinite(chunk_embeddings).all(axis=1)
        | (np.linalg.norm(chunk_embeddings, axis=1) < 1e-8)
    )
    print(
        f"Indexed {chunk_embeddings.shape[0]} chunks → {chunk_embeddings.shape[1]}d vectors"
        + (f" ({n_bad} chunks with bad embeddings will be ignored)" if n_bad else "")
    )

    # Interactive loop
    print("\nReady! Type a question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        try:
            answer = ask(
                question,
                chunks,
                chunk_embeddings,
                args.encoderfile_url,
                args.llamafile_url,
                args.system_prompt,
                args.top_k,
            )
            print(f"\n{answer}\n")
        except requests.exceptions.ConnectionError:
            print(f"\nError: Could not connect to llamafile at {args.llamafile_url}.")
            print("Make sure your llamafile server is running.\n")
        except requests.RequestException as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
