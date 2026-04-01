"""
CVE Semantic Search POC
========================
Encoderfile + Qdrant integration demo.

Demonstrates a fully local, privacy-first vulnerability search system:
  - Encoderfile generates embeddings (no cloud API, no Python ML runtime)
  - Qdrant stores and searches vectors (self-hosted, no data leaves the network)

Usage:
    python main.py                      # Index CVEs, run one example query
    python main.py --interactive        # Index CVEs, then interactive search
    python main.py -q "your query"      # Index CVEs, run a specific query
"""

import argparse
import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from embedders import get_embedder

with open("cve_data.json") as f:
    SAMPLE_CVES = json.load(f)

COLLECTION = "cve-reports"


def build_index(client: QdrantClient, embedder) -> None:
    """Embed all CVEs and store them in Qdrant."""

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embedder.dimension,
            distance=Distance.COSINE,
        ),
    )

    descriptions = [cve["description"] for cve in SAMPLE_CVES]
    print(f"Embedding {len(descriptions)} CVE descriptions...")
    vectors = embedder.embed_batch(descriptions)

    points = []
    for i, (cve, vector) in enumerate(zip(SAMPLE_CVES, vectors)):
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "cve_id": cve["id"],
                    "severity": cve["severity"],
                    "description": cve["description"],
                    "affected": cve["affected"],
                    "category": cve["category"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} CVEs into Qdrant.\n")


def search(
    client: QdrantClient,
    embedder,
    query: str,
    top_k: int = 5,
    severity: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """
    Semantic search over CVE reports.

    Args:
        query: Natural language search (e.g. "authentication bypass in VPN")
        top_k: Number of results to return
        severity: Optional filter (CRITICAL, HIGH, MEDIUM)
        category: Optional filter (remote-code-execution, authentication-bypass, etc.)
    """
    query_vector = embedder.embed(query)

    conditions = []
    if severity:
        conditions.append(
            FieldCondition(key="severity", match=MatchValue(value=severity.upper()))
        )
    if category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )

    search_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    formatted = []
    for point in results.points:
        formatted.append(
            {
                "cve_id": point.payload["cve_id"],
                "score": round(point.score, 4),
                "severity": point.payload["severity"],
                "category": point.payload["category"],
                "description": point.payload["description"],
                "affected": point.payload["affected"],
            }
        )

    return formatted


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print search results."""
    print(f'  Query: "{query}"')
    print(f"  Results: {len(results)}")
    print(f"  {'—' * 60}")

    severity_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}

    for i, r in enumerate(results, 1):
        icon = severity_icons.get(r["severity"], "⚪")
        print(
            f"\n  {i}. {icon} {r['cve_id']}  (score: {r['score']})  [{r['severity']}]"
        )
        print(f"     Category: {r['category']}")
        print(f"     Affected: {r['affected']}")
        desc = r["description"]
        if len(desc) > 200:
            desc = desc[:200] + "..."
        print(f"     {desc}")

    print()


def interactive_loop(client: QdrantClient, embedder) -> None:
    """Interactive search mode."""
    print("Interactive search (type 'quit' to exit)\n")

    while True:
        try:
            query = input("  Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        results = search(client, embedder, query, top_k=5)
        print()
        print_results(query, results)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="CVE Semantic Search — Encoderfile + Qdrant"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Drop into interactive search after indexing",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Run a specific query instead of the default example",
    )
    args = parser.parse_args()

    # Setup
    client = QdrantClient(":memory:")
    embedder = get_embedder(prefer_encoderfile=True)

    # Ingest
    build_index(client, embedder)

    # Run one query to prove it works
    query = args.query or "authentication bypass in VPN products"
    results = search(client, embedder, query, top_k=5)
    print_results(query, results)

    # Interactive mode if requested
    if args.interactive:
        interactive_loop(client, embedder)


if __name__ == "__main__":
    main()
