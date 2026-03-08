#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec


DEFAULT_MODEL = Path("data/acfun.videoinfo.20260307.full.word2vec.kv")
DEFAULT_OUTPUT_DIR = Path("data/neo4j_word2vec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Word2Vec similarity graph to Neo4j-compatible CSV files."
    )
    parser.add_argument("terms", nargs="*", help="Optional seed terms for a focused subgraph")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to .kv or .model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for nodes.csv and edges.csv",
    )
    parser.add_argument("--topn", type=int, default=10, help="Neighbors per term")
    parser.add_argument("--min-score", type=float, default=0.7, help="Minimum cosine similarity")
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Neighbor expansion depth when seed terms are provided",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=5_000,
        help="Maximum vocabulary size to export when no seed terms are provided",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Keep source->target edges instead of deduplicating as undirected pairs",
    )
    return parser.parse_args()


def load_vectors(path: Path) -> KeyedVectors:
    if path.suffix == ".model":
        return Word2Vec.load(path.as_posix()).wv
    return KeyedVectors.load(path.as_posix(), mmap="r")


def get_term_count(vectors: KeyedVectors, term: str) -> int | None:
    try:
        return int(vectors.get_vecattr(term, "count"))
    except (KeyError, ValueError, AttributeError):
        return None


def iter_seed_subgraph(
    vectors: KeyedVectors,
    seeds: list[str],
    depth: int,
    topn: int,
    min_score: float,
) -> tuple[dict[str, dict[str, int | bool | None]], list[dict[str, str | int | float]]]:
    nodes: dict[str, dict[str, int | bool | None]] = {}
    edges: list[dict[str, str | int | float]] = []
    seen_depth: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for term in seeds:
        if term not in vectors:
            continue
        queue.append((term, 0))
        seen_depth[term] = 0
        nodes[term] = {"freq": get_term_count(vectors, term), "is_seed": True}

    while queue:
        term, current_depth = queue.popleft()
        neighbors = vectors.most_similar(term, topn=topn)
        rank = 0
        for neighbor, raw_score in neighbors:
            score = float(raw_score)
            if score < min_score:
                continue
            rank += 1
            nodes.setdefault(
                neighbor,
                {"freq": get_term_count(vectors, neighbor), "is_seed": neighbor in seeds},
            )
            edges.append(
                {
                    "source": term,
                    "target": neighbor,
                    "score": score,
                    "rank": rank,
                }
            )
            if current_depth >= depth:
                continue
            next_depth = current_depth + 1
            previous_depth = seen_depth.get(neighbor)
            if previous_depth is None or next_depth < previous_depth:
                seen_depth[neighbor] = next_depth
                queue.append((neighbor, next_depth))

    return nodes, edges


def iter_vocab_graph(
    vectors: KeyedVectors,
    max_vocab: int,
    topn: int,
    min_score: float,
) -> tuple[dict[str, dict[str, int | bool | None]], list[dict[str, str | int | float]]]:
    nodes: dict[str, dict[str, int | bool | None]] = {}
    edges: list[dict[str, str | int | float]] = []

    for term in vectors.index_to_key[:max_vocab]:
        nodes.setdefault(term, {"freq": get_term_count(vectors, term), "is_seed": False})
        rank = 0
        for neighbor, raw_score in vectors.most_similar(term, topn=topn):
            score = float(raw_score)
            if score < min_score:
                continue
            rank += 1
            nodes.setdefault(
                neighbor,
                {"freq": get_term_count(vectors, neighbor), "is_seed": False},
            )
            edges.append(
                {
                    "source": term,
                    "target": neighbor,
                    "score": score,
                    "rank": rank,
                }
            )

    return nodes, edges


def dedupe_edges(
    edges: list[dict[str, str | int | float]],
    directed: bool,
) -> list[dict[str, str | int | float]]:
    if directed:
        return edges

    deduped: dict[tuple[str, str], dict[str, str | int | float]] = {}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        key = tuple(sorted((source, target)))
        existing = deduped.get(key)
        if existing is None or float(edge["score"]) > float(existing["score"]):
            deduped[key] = {
                "source": key[0],
                "target": key[1],
                "score": edge["score"],
                "rank": edge["rank"],
            }
    return list(deduped.values())


def write_nodes(path: Path, nodes: dict[str, dict[str, int | bool | None]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "freq", "is_seed"])
        for name in sorted(nodes):
            attrs = nodes[name]
            writer.writerow([name, attrs["freq"], str(bool(attrs["is_seed"])).lower()])


def write_edges(path: Path, edges: list[dict[str, str | int | float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "score", "rank", "model"])
        for edge in sorted(edges, key=lambda item: (str(item["source"]), -float(item["score"]), str(item["target"]))):
            writer.writerow(
                [
                    edge["source"],
                    edge["target"],
                    f"{float(edge['score']):.6f}",
                    edge["rank"],
                    "word2vec",
                ]
            )


def main() -> None:
    args = parse_args()
    vectors = load_vectors(args.model)

    if args.terms:
        nodes, raw_edges = iter_seed_subgraph(
            vectors=vectors,
            seeds=args.terms,
            depth=args.depth,
            topn=args.topn,
            min_score=args.min_score,
        )
    else:
        nodes, raw_edges = iter_vocab_graph(
            vectors=vectors,
            max_vocab=args.max_vocab,
            topn=args.topn,
            min_score=args.min_score,
        )

    edges = dedupe_edges(raw_edges, directed=args.directed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.output_dir / "nodes.csv"
    edges_path = args.output_dir / "edges.csv"
    write_nodes(nodes_path, nodes)
    write_edges(edges_path, edges)

    print(f"model: {args.model}")
    print(f"nodes: {nodes_path} ({len(nodes)})")
    print(f"edges: {edges_path} ({len(edges)})")


if __name__ == "__main__":
    main()
