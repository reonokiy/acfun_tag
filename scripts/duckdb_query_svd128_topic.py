#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


DEFAULT_METADATA = Path("data/acfun.videoinfo.20260307.full.flattened.parquet")
DEFAULT_SVD = Path("data/acfun.videoinfo.20260307.full.tfidf.svd128.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search flattened metadata with DuckDB, build a seed centroid from "
            "svd128 vectors, score all rows by cosine similarity, and write hits."
        )
    )
    parser.add_argument("query", help="Keyword or phrase used to bootstrap the topic")
    parser.add_argument("output", type=Path, help="Output parquet path for ranked hits")
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
    parser.add_argument("--top-k", type=int, default=1000, help="How many hits to keep")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum cosine similarity")
    parser.add_argument(
        "--exclude-seeds",
        action="store_true",
        help="Exclude keyword-matched seed rows from final ranking",
    )
    parser.add_argument(
        "--show-seeds",
        type=int,
        default=5,
        help="How many nearest seed rows to print",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help="Expected SVD vector length",
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=None,
        help="Optional parquet path for matched seed rows",
    )
    parser.add_argument(
        "--centroid-output",
        type=Path,
        default=None,
        help="Optional parquet path for the centroid row",
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
            payload_user_name AS user_name,
            payload_viewCount AS view_count,
            payload_likeCount AS like_count,
            payload_commentCount AS comment_count,
            payload_createTime AS create_time,
            payload_coverUrl AS cover_url,
            payload_dougaId AS douga_id,
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

    filtered_sql = f"""
        SELECT
            id,
            title,
            description,
            parent_channel,
            channel,
            tags,
            user_name,
            view_count,
            like_count,
            comment_count,
            create_time,
            cover_url,
            douga_id
        FROM ({metadata_base_sql()}) metadata
        WHERE {match_sql}
    """
    filtered_params: list[object] = [args.metadata.as_posix(), *match_params]
    if args.seed_limit is not None:
        filtered_sql += " LIMIT ?"
        filtered_params.append(args.seed_limit)

    seeds = con.execute(filtered_sql, filtered_params).fetch_arrow_table()
    con.register("matched_seeds", seeds)

    seed_count = seeds.num_rows
    print(
        json.dumps(
            {
                "query": args.query,
                "phrase": phrase,
                "term_mode": args.term_mode,
                "seed_limit": args.seed_limit,
                "seed_count": seed_count,
                "top_k": args.top_k,
                "min_score": args.min_score,
                "exclude_seeds": args.exclude_seeds,
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
        pq.write_table(seeds, args.seed_output)

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

    centroid = con.execute(
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
    ).fetchone()[0]
    if centroid is None:
        raise ValueError("Failed to compute centroid")

    print(
        json.dumps(
            {
                "centroid_dimensions": len(centroid),
                "centroid": centroid,
            },
            ensure_ascii=False,
        ),
        flush=True,
        )

    if args.show_seeds > 0:
        seed_preview = con.execute(
            """
            SELECT
                m.id,
                m.title,
                list_cosine_similarity(v.vector, ?::FLOAT[]) AS score
            FROM matched_seeds m
            JOIN matched_vectors v USING (id)
            ORDER BY score DESC, m.id ASC
            LIMIT ?
            """,
            [centroid, args.show_seeds],
        ).fetch_arrow_table()
        print(
            json.dumps(
                {
                    "nearest_seed_count": seed_preview.num_rows,
                    "nearest_seeds": [
                        {"id": row["id"], "title": row["title"]}
                        for row in seed_preview.to_pylist()
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.centroid_output is not None:
        args.centroid_output.parent.mkdir(parents=True, exist_ok=True)
        if args.centroid_output.exists():
            args.centroid_output.unlink()
        centroid_rel = con.execute(
            """
            SELECT
                ? AS query,
                ? AS phrase,
                ? AS term_mode,
                ? AS seed_count,
                ?::FLOAT[] AS vector
            """,
            [args.query, phrase, args.term_mode, vector_count, centroid],
        ).fetch_arrow_table()
        pq.write_table(centroid_rel, args.centroid_output)

    exclude_join = ""
    exclude_filter = ""
    if args.exclude_seeds:
        exclude_join = "LEFT JOIN matched_seeds seed USING (id)"
        exclude_filter = "AND seed.id IS NULL"

    result = con.execute(
        f"""
        WITH scored AS (
            SELECT
                s.id,
                list_cosine_similarity(s.vector, ?::FLOAT[]) AS score
            FROM read_parquet(?) s
            {exclude_join}
            WHERE true
            {exclude_filter}
        ),
        metadata AS (
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
                payload_user_name AS user_name,
                payload_viewCount AS view_count,
                payload_likeCount AS like_count,
                payload_commentCount AS comment_count,
                payload_createTime AS create_time,
                payload_coverUrl AS cover_url,
                payload_dougaId AS douga_id
            FROM read_parquet(?)
        )
        SELECT
            ? AS query,
            ? AS phrase,
            ? AS term_mode,
            ? AS seed_count,
            scored.id,
            scored.score,
            metadata.title,
            metadata.description,
            metadata.parent_channel,
            metadata.channel,
            metadata.tags,
            metadata.user_name,
            metadata.view_count,
            metadata.like_count,
            metadata.comment_count,
            metadata.create_time,
            metadata.cover_url,
            metadata.douga_id
        FROM scored
        JOIN metadata USING (id)
        WHERE scored.score >= ?
        ORDER BY scored.score DESC, scored.id ASC
        LIMIT ?
        """,
        [
            centroid,
            args.svd.as_posix(),
            args.metadata.as_posix(),
            args.query,
            phrase,
            args.term_mode,
            vector_count,
            args.min_score,
            args.top_k,
        ],
    ).fetch_arrow_table()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    pq.write_table(result, args.output)

    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "rows": result.num_rows,
                "seed_count": vector_count,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
