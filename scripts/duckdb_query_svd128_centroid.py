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
            "Read a centroid parquet, score all svd128 vectors with DuckDB cosine "
            "similarity, join metadata, and write ranked parquet results."
        )
    )
    parser.add_argument("centroid", type=Path, help="Centroid parquet path")
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
        "--seed-parquet",
        type=Path,
        default=None,
        help="Optional seed parquet path used for --exclude-seeds",
    )
    parser.add_argument("--top-k", type=int, default=1000, help="How many hits to keep")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum cosine similarity")
    parser.add_argument(
        "--exclude-seeds",
        action="store_true",
        help="Exclude rows present in --seed-parquet from final ranking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.exclude_seeds and args.seed_parquet is None:
        raise ValueError("--exclude-seeds requires --seed-parquet")

    con = duckdb.connect()

    exclude_join = ""
    exclude_filter = ""
    params: list[object] = [
        args.svd.as_posix(),
        args.centroid.as_posix(),
    ]
    if args.exclude_seeds:
        exclude_join = "LEFT JOIN read_parquet(?) seed USING (id)"
        exclude_filter = "AND seed.id IS NULL"
        params.append(args.seed_parquet.as_posix())
    params.extend(
        [
            args.metadata.as_posix(),
            args.min_score,
            args.top_k,
        ]
    )

    query = f"""
        WITH centroid AS (
            SELECT
                query,
                phrase,
                term_mode,
                seed_count,
                vector
            FROM read_parquet(?)
        ),
        scored AS (
            SELECT
                s.id,
                c.query,
                c.phrase,
                c.term_mode,
                c.seed_count,
                list_cosine_similarity(s.vector, c.vector) AS score
            FROM read_parquet(?) s
            CROSS JOIN centroid c
            {exclude_join}
            WHERE true
            {exclude_filter}
        )
        SELECT
            scored.query,
            scored.phrase,
            scored.term_mode,
            scored.seed_count,
            scored.id,
            scored.score,
            f.payload_title AS title,
            f.payload_description AS description,
            f.payload_channel_parentName AS parent_channel,
            f.payload_channel_name AS channel,
            coalesce(
                array_to_string(
                    list_transform(f.payload_tagList, x -> coalesce(x.name, '')),
                    ' '
                ),
                ''
            ) AS tags,
            f.payload_user_name AS user_name,
            f.payload_viewCount AS view_count,
            f.payload_likeCount AS like_count,
            f.payload_commentCount AS comment_count,
            f.payload_createTime AS create_time,
            f.payload_coverUrl AS cover_url,
            f.payload_dougaId AS douga_id
        FROM scored
        JOIN read_parquet(?) f USING (id)
        WHERE scored.score >= ?
        ORDER BY scored.score DESC, scored.id ASC
        LIMIT ?
    """

    # Parameter order follows the query text.
    if args.exclude_seeds:
        params = [
            args.centroid.as_posix(),
            args.svd.as_posix(),
            args.seed_parquet.as_posix(),
            args.metadata.as_posix(),
            args.min_score,
            args.top_k,
        ]
    else:
        params = [
            args.centroid.as_posix(),
            args.svd.as_posix(),
            args.metadata.as_posix(),
            args.min_score,
            args.top_k,
        ]

    result = con.execute(query, params).fetch_arrow_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    pq.write_table(result, args.output)

    print(
        json.dumps(
            {
                "centroid": args.centroid.as_posix(),
                "output": args.output.as_posix(),
                "rows": result.num_rows,
                "top_k": args.top_k,
                "min_score": args.min_score,
                "exclude_seeds": args.exclude_seeds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
