#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_METADATA = Path("data/acfun.videoinfo.20260307.full.flattened.parquet")
DEFAULT_SVD = Path("data/acfun.videoinfo.20260307.full.tfidf.svd128.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find keyword-matched seed rows from flattened metadata, join them "
            "with svd128 vectors in DuckDB, and write the centroid as parquet."
        )
    )
    parser.add_argument("query", help="Keyword or phrase used to bootstrap the topic")
    parser.add_argument("centroid_output", type=Path, help="Output parquet path for the centroid row")
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=None,
        help="Optional parquet path for matched seed rows",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Flattened metadata parquet path",
    )
    parser.add_argument(
        "--svd",
        type=Path,
        default=DEFAULT_SVD,
        help="SVD parquet path",
    )
    parser.add_argument(
        "--term-mode",
        choices=("phrase", "all", "any"),
        default="phrase",
        help="How to match the query against metadata text",
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=None,
        help="Optional limit on matched seed rows before centroid aggregation",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help="Expected SVD vector length",
    )
    return parser.parse_args()


def normalize_query(text: str) -> tuple[str, list[str]]:
    phrase = " ".join(text.strip().lower().split())
    terms = [term for term in phrase.split(" ") if term]
    return phrase, terms


def build_match_clause(term_mode: str, terms: list[str]) -> tuple[str, list[str]]:
    if not terms:
        raise ValueError("Query must contain at least one non-space token")
    if term_mode == "phrase":
        return "haystack LIKE ?", [f"%{' '.join(terms)}%"]
    joiner = " AND " if term_mode == "all" else " OR "
    return joiner.join(["haystack LIKE ?"] * len(terms)), [f"%{term}%" for term in terms]


def metadata_base_sql() -> str:
    return """
        SELECT
            id,
            payload_title AS title,
            payload_description AS description,
            payload_channel_parentName AS parent_channel,
            payload_channel_name AS channel,
            coalesce(
                array_to_string(
                    list_transform(payload_tagList, x -> coalesce(x.name, '')),
                    ' '
                ),
                ''
            ) AS tags,
            lower(
                concat_ws(
                    '\n',
                    coalesce(payload_title, ''),
                    coalesce(payload_description, ''),
                    coalesce(payload_channel_parentName, ''),
                    coalesce(payload_channel_name, ''),
                    coalesce(
                        array_to_string(
                            list_transform(payload_tagList, x -> coalesce(x.name, '')),
                            ' '
                        ),
                        ''
                    )
                )
            ) AS haystack
        FROM read_parquet(?)
    """


def sql_quote(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    phrase, terms = normalize_query(args.query)
    match_sql, match_params = build_match_clause(args.term_mode, terms)

    con = duckdb.connect()
    base_sql = metadata_base_sql()

    filtered_sql = f"""
        SELECT id, title, description, parent_channel, channel, tags
        FROM ({base_sql}) metadata
        WHERE {match_sql}
    """
    filtered_params: list[object] = [args.metadata.as_posix(), *match_params]
    if args.seed_limit is not None:
        filtered_sql += " LIMIT ?"
        filtered_params.append(args.seed_limit)

    seeds_table = con.execute(filtered_sql, filtered_params).fetch_arrow_table()
    con.register("matched_seeds", seeds_table)

    seed_count = seeds_table.num_rows
    print(
        json.dumps(
            {
                "query": args.query,
                "phrase": phrase,
                "term_mode": args.term_mode,
                "seed_limit": args.seed_limit,
                "seed_count": seed_count,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if seed_count == 0:
        raise ValueError("No seed rows matched the query")

    if args.seed_output is not None:
        args.seed_output.parent.mkdir(parents=True, exist_ok=True)
        if args.seed_output.exists():
            args.seed_output.unlink()
        pq.write_table(seeds_table, args.seed_output)
        print(f"[seed] wrote rows={seed_count} path={args.seed_output}", flush=True)

    svd_path = sql_quote(args.svd)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW matched_vectors AS
        SELECT s.id, s.vector
        FROM read_parquet('{svd_path}') s
        JOIN matched_seeds m USING (id)
        """
    )

    vector_count = con.execute("SELECT count(*) FROM matched_vectors").fetchone()[0]
    if vector_count == 0:
        raise ValueError("Seed rows matched metadata, but none were found in the SVD parquet")

    centroid_row = con.execute(
        """
        WITH idx AS (
            SELECT generate_series AS idx
            FROM generate_series(1, ?)
        ),
        per_dim AS (
            SELECT
                idx,
                avg(list_extract(vector, idx))::FLOAT AS value
            FROM matched_vectors
            CROSS JOIN idx
            GROUP BY idx
        )
        SELECT list(value ORDER BY idx) AS centroid
        FROM per_dim
        """,
        [args.dimensions],
    ).fetchone()
    centroid = centroid_row[0]
    if centroid is None:
        raise ValueError("Failed to compute centroid")

    args.centroid_output.parent.mkdir(parents=True, exist_ok=True)
    if args.centroid_output.exists():
        args.centroid_output.unlink()

    centroid_table = pa.table(
        {
            "query": [args.query],
            "phrase": [phrase],
            "term_mode": [args.term_mode],
            "seed_count": [vector_count],
            "vector": pa.array([centroid], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(centroid_table, args.centroid_output)
    print(
        f"[centroid] wrote seeds={vector_count} dims={len(centroid)} path={args.centroid_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
