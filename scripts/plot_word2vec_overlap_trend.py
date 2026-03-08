#!/usr/bin/env python3
import argparse
import csv
import json
import logging
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt

from duckdb_bm25_search import DEFAULT_DB, DEFAULT_TABLE, normalize_query_terms
from query_word2vec_terms import DEFAULT_MODEL, load_vectors


DEFAULT_OUTPUT = Path("data/plots/word2vec_overlap_trend.png")
DEFAULT_CSV_OUTPUT = Path("data/plots/word2vec_overlap_trend.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot overlap trend between seed-term search and top-k related-only Word2Vec searches."
    )
    parser.add_argument("term", help="Seed term, for example: 鬼畜")
    parser.add_argument("--plot-term-label", default=None, help="ASCII label shown in the plot title")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to .kv or .model")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB BM25 index path")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="DuckDB table name")
    parser.add_argument("--max-topk", type=int, default=20, help="Maximum related terms to include")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum cosine similarity")
    parser.add_argument("--tokenize-query", action="store_true", help="Tokenize query terms with the BM25 tokenizer")
    parser.add_argument("--conjunctive", action="store_true", help="Require all terms within a field")
    parser.add_argument("--title-weight", type=float, default=3.0, help="Weight for title score")
    parser.add_argument("--description-weight", type=float, default=1.0, help="Weight for description score")
    parser.add_argument("--tags-weight", type=float, default=4.0, help="Weight for tags score")
    parser.add_argument("--parent-weight", type=float, default=1.5, help="Weight for parent partition score")
    parser.add_argument("--channel-weight", type=float, default=2.0, help="Weight for channel partition score")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output figure path")
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT, help="Output CSV path")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level",
    )
    return parser.parse_args()


def get_related_terms(term: str, model_path: Path, max_topk: int, min_score: float) -> list[tuple[str, float]]:
    vectors = load_vectors(model_path)
    if term not in vectors:
        return []
    return [
        (token, float(score))
        for token, score in vectors.most_similar(term, topn=max_topk)
        if float(score) >= min_score
    ]


def fetch_match_ids(
    con: duckdb.DuckDBPyConnection,
    table: str,
    query_terms: list[str],
    conjunctive: bool,
    title_weight: float,
    description_weight: float,
    tags_weight: float,
    parent_weight: float,
    channel_weight: float,
) -> set[int]:
    query_text = " ".join(query_terms)
    fts_schema = f"fts_main_{table}"
    sql = f"""
        WITH scored AS (
            SELECT
                id,
                {fts_schema}.match_bm25(id, ?, fields := 'title_terms', conjunctive := ?) AS title_score,
                {fts_schema}.match_bm25(id, ?, fields := 'description_terms', conjunctive := ?) AS description_score,
                {fts_schema}.match_bm25(id, ?, fields := 'tags_terms', conjunctive := ?) AS tags_score,
                {fts_schema}.match_bm25(id, ?, fields := 'parent_terms', conjunctive := ?) AS parent_score,
                {fts_schema}.match_bm25(id, ?, fields := 'channel_terms', conjunctive := ?) AS channel_score
            FROM {table}
        )
        SELECT id
        FROM scored
        WHERE (
            coalesce(title_score, 0) * ?
            + coalesce(description_score, 0) * ?
            + coalesce(tags_score, 0) * ?
            + coalesce(parent_score, 0) * ?
            + coalesce(channel_score, 0) * ?
        ) > 0
    """
    bool_flag = 1 if conjunctive else 0
    params: list[object] = [
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        title_weight,
        description_weight,
        tags_weight,
        parent_weight,
        channel_weight,
    ]
    return {int(row[0]) for row in con.execute(sql, params).fetchall()}


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    related_terms = get_related_terms(args.term, args.model, args.max_topk, args.min_score)
    con = duckdb.connect(args.db.as_posix(), read_only=True)
    con.execute("LOAD fts")

    seed_query_terms = normalize_query_terms([args.term], args.tokenize_query)
    seed_ids = fetch_match_ids(
        con=con,
        table=args.table,
        query_terms=seed_query_terms,
        conjunctive=args.conjunctive,
        title_weight=args.title_weight,
        description_weight=args.description_weight,
        tags_weight=args.tags_weight,
        parent_weight=args.parent_weight,
        channel_weight=args.channel_weight,
    )
    seed_count = len(seed_ids)

    rows: list[dict[str, object]] = []
    for topk in range(1, len(related_terms) + 1):
        current_terms = [token for token, _ in related_terms[:topk]]
        normalized_terms = normalize_query_terms(current_terms, args.tokenize_query)
        related_ids = fetch_match_ids(
            con=con,
            table=args.table,
            query_terms=normalized_terms,
            conjunctive=args.conjunctive,
            title_weight=args.title_weight,
            description_weight=args.description_weight,
            tags_weight=args.tags_weight,
            parent_weight=args.parent_weight,
            channel_weight=args.channel_weight,
        )
        intersection = seed_ids & related_ids
        union_size = len(seed_ids | related_ids)
        overlap_ratio = 0.0 if seed_count == 0 else len(intersection) / seed_count
        jaccard = 0.0 if union_size == 0 else len(intersection) / union_size
        added_term, added_score = related_terms[topk - 1]
        rows.append(
            {
                "seed_term": args.term,
                "topk": topk,
                "query_text": " ".join(normalized_terms),
                "added_term": added_term,
                "added_score": added_score,
                "seed_hit_count": seed_count,
                "related_hit_count": len(related_ids),
                "intersection_count": len(intersection),
                "union_count": union_size,
                "overlap_ratio": overlap_ratio,
                "jaccard_ratio": jaccard,
            }
        )

    con.close()
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed_term",
                "topk",
                "added_term",
                "added_score",
                "seed_hit_count",
                "related_hit_count",
                "intersection_count",
                "union_count",
                "overlap_ratio",
                "jaccard_ratio",
                "query_text",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(rows: list[dict[str, object]], output: Path, dpi: int, plot_term_label: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    term_label = plot_term_label or "seed-term"
    xs = [int(row["topk"]) for row in rows]
    ys = [float(row["overlap_ratio"]) * 100.0 for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o", linewidth=2, color="#1d4ed8", label="Overlap ratio vs seed search")
    for x, row, y in zip(xs, rows, ys, strict=True):
        plt.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1e3a8a",
        )
    plt.xticks(xs)
    plt.xlabel("Related-only top-k")
    plt.ylabel("Overlap ratio with seed search (%)")
    plt.title(f"Related-only overlap trend for '{term_label}'")
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output.as_posix(), dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rows = build_rows(args)
    if not rows:
        raise RuntimeError("No related terms available to build overlap trend")

    write_csv(rows, args.csv_output)
    plot_rows(rows, args.output, args.dpi, args.plot_term_label)

    best_overlap = max(rows, key=lambda row: float(row["overlap_ratio"]))
    print(
        json.dumps(
            {
                "term": args.term,
                "points": len(rows),
                "seed_hit_count": rows[0]["seed_hit_count"],
                "best_topk": best_overlap["topk"],
                "best_overlap_ratio": best_overlap["overlap_ratio"],
                "best_intersection_count": best_overlap["intersection_count"],
                "csv_output": args.csv_output.as_posix(),
                "figure_output": args.output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
