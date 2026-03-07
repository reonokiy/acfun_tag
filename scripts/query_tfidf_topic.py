#!/usr/bin/env python3
import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


DEFAULT_META_COLUMNS = {
    "id": "id",
    "title": "payload_title",
    "description": "payload_description",
    "channel": "payload_channel_name",
    "parent_channel": "payload_channel_parentName",
    "tags": "payload_tagList",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query a TF-IDF parquet by first finding keyword seed documents in "
            "metadata, then retrieving similar items using cosine similarity."
        )
    )
    parser.add_argument("query", help="Keyword or phrase to bootstrap the topic")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/acfun.videoinfo.20260307.full.flattened.parquet"),
        help="Metadata parquet path",
    )
    parser.add_argument(
        "--tfidf",
        type=Path,
        default=Path("data/acfun.videoinfo.20260307.full.tfidf.parquet"),
        help="TF-IDF parquet path",
    )
    parser.add_argument("--batch-size", type=int, default=50_000, help="Rows per scan batch")
    parser.add_argument("--seed-limit", type=int, default=20_000, help="Maximum seed documents to use")
    parser.add_argument("--top-k", type=int, default=100, help="Number of final hits to keep")
    parser.add_argument("--min-score", type=float, default=0.15, help="Minimum cosine similarity")
    parser.add_argument(
        "--term-mode",
        choices=("phrase", "all", "any"),
        default="phrase",
        help="How to match the query against metadata text",
    )
    parser.add_argument(
        "--exclude-seeds",
        action="store_true",
        help="Exclude keyword-matched seed items from the final ranking",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional parquet path for ranked results",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many hits to print",
    )
    return parser.parse_args()


def normalize_query(text: str) -> tuple[str, list[str]]:
    phrase = " ".join(text.strip().lower().split())
    terms = [term for term in phrase.split(" ") if term]
    return phrase, terms


def tags_to_text(tag_value) -> str:
    if not tag_value:
        return ""
    parts: list[str] = []
    for item in tag_value:
        if isinstance(item, dict):
            name = item.get("name")
        elif item is None:
            name = None
        else:
            name = getattr(item, "get", lambda *_: None)("name")
        if name:
            parts.append(str(name))
    return " ".join(parts)


def row_matches(text: str, phrase: str, terms: list[str], mode: str) -> bool:
    if not text:
        return False
    if mode == "phrase":
        return phrase in text
    if mode == "all":
        return all(term in text for term in terms)
    return any(term in text for term in terms)


def build_haystack(title, description, channel, parent_channel, tag_list) -> str:
    return " \n ".join(
        [
            (title or ""),
            (description or ""),
            (channel or ""),
            (parent_channel or ""),
            tags_to_text(tag_list),
        ]
    ).lower()


def find_seed_ids(
    metadata_path: Path,
    query_phrase: str,
    terms: list[str],
    term_mode: str,
    batch_size: int,
    seed_limit: int,
) -> tuple[set[int], dict[int, dict], int]:
    dataset = ds.dataset(metadata_path.as_posix(), format="parquet")
    columns = [
        DEFAULT_META_COLUMNS["id"],
        DEFAULT_META_COLUMNS["title"],
        DEFAULT_META_COLUMNS["description"],
        DEFAULT_META_COLUMNS["channel"],
        DEFAULT_META_COLUMNS["parent_channel"],
        DEFAULT_META_COLUMNS["tags"],
    ]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)

    seed_ids: set[int] = set()
    seed_meta: dict[int, dict] = {}
    scanned = 0

    for batch_index, batch in enumerate(scanner.to_batches()):
        ids = batch.column(0).cast(pa.int64()).to_pylist()
        titles = batch.column(1).to_pylist()
        descriptions = batch.column(2).to_pylist()
        channels = batch.column(3).to_pylist()
        parent_channels = batch.column(4).to_pylist()
        tags = batch.column(5).to_pylist()

        for doc_id, title, description, channel, parent_channel, tag_list in zip(
            ids,
            titles,
            descriptions,
            channels,
            parent_channels,
            tags,
            strict=True,
        ):
            haystack = build_haystack(title, description, channel, parent_channel, tag_list)
            if not row_matches(haystack, query_phrase, terms, term_mode):
                continue
            seed_ids.add(int(doc_id))
            seed_meta[int(doc_id)] = {
                "title": title,
                "description": description,
                "channel": channel,
                "parent_channel": parent_channel,
                "tags": tags_to_text(tag_list),
            }
            if len(seed_ids) >= seed_limit:
                scanned += len(ids)
                return seed_ids, seed_meta, scanned

        scanned += len(ids)
        if (batch_index + 1) % 20 == 0:
            print(f"[seed] batches={batch_index + 1} scanned={scanned} matched={len(seed_ids)}", flush=True)

    return seed_ids, seed_meta, scanned


def accumulate_centroid(tfidf_path: Path, seed_ids: set[int], batch_size: int) -> tuple[np.ndarray, int]:
    dataset = ds.dataset(tfidf_path.as_posix(), format="parquet")
    scanner = dataset.scanner(columns=["id", "vector"], batch_size=batch_size, use_threads=True)

    centroid: np.ndarray | None = None
    used = 0

    for batch_index, batch in enumerate(scanner.to_batches()):
        ids = batch.column(0).cast(pa.int64()).to_pylist()
        vector_col = batch.column(1)
        indices_rows = vector_col.field("indices").to_pylist()
        values_rows = vector_col.field("values").to_pylist()

        for doc_id, indices, values in zip(ids, indices_rows, values_rows, strict=True):
            if int(doc_id) not in seed_ids:
                continue
            if centroid is None:
                feature_count = max((max(indices) + 1) if indices else 1, 1)
                centroid = np.zeros(feature_count, dtype=np.float32)
            if indices:
                max_index = max(indices)
                if max_index >= centroid.shape[0]:
                    grown = np.zeros(max_index + 1, dtype=np.float32)
                    grown[: centroid.shape[0]] = centroid
                    centroid = grown
                centroid[np.asarray(indices, dtype=np.int32)] += np.asarray(values, dtype=np.float32)
            used += 1

        if (batch_index + 1) % 20 == 0:
            print(f"[centroid] batches={batch_index + 1} matched={used}", flush=True)

    if centroid is None or used == 0:
        raise ValueError("No seed vectors found in TF-IDF parquet")

    centroid /= float(used)
    norm = np.linalg.norm(centroid)
    if norm == 0.0:
        raise ValueError("Centroid vector is zero")
    centroid /= norm
    return centroid, used


def score_all(
    tfidf_path: Path,
    centroid: np.ndarray,
    batch_size: int,
    top_k: int,
    min_score: float,
    seed_ids: set[int],
    exclude_seeds: bool,
) -> list[tuple[float, int]]:
    dataset = ds.dataset(tfidf_path.as_posix(), format="parquet")
    scanner = dataset.scanner(columns=["id", "vector"], batch_size=batch_size, use_threads=True)
    heap: list[tuple[float, int]] = []
    seen = 0

    for batch_index, batch in enumerate(scanner.to_batches()):
        ids = batch.column(0).cast(pa.int64()).to_pylist()
        vector_col = batch.column(1)
        indices_rows = vector_col.field("indices").to_pylist()
        values_rows = vector_col.field("values").to_pylist()

        for doc_id, indices, values in zip(ids, indices_rows, values_rows, strict=True):
            doc_id = int(doc_id)
            if exclude_seeds and doc_id in seed_ids:
                continue
            if not indices:
                continue
            idx = np.asarray(indices, dtype=np.int32)
            vals = np.asarray(values, dtype=np.float32)
            score = float(np.dot(centroid[idx], vals))
            if score < min_score:
                continue
            item = (score, doc_id)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)
            seen += 1

        if (batch_index + 1) % 20 == 0:
            print(f"[score] batches={batch_index + 1} candidates={seen}", flush=True)

    return sorted(heap, reverse=True)


def fetch_result_metadata(metadata_path: Path, ids: set[int], batch_size: int) -> dict[int, dict]:
    dataset = ds.dataset(metadata_path.as_posix(), format="parquet")
    columns = [
        DEFAULT_META_COLUMNS["id"],
        DEFAULT_META_COLUMNS["title"],
        DEFAULT_META_COLUMNS["description"],
        DEFAULT_META_COLUMNS["channel"],
        DEFAULT_META_COLUMNS["parent_channel"],
        DEFAULT_META_COLUMNS["tags"],
    ]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    out: dict[int, dict] = {}

    for batch in scanner.to_batches():
        batch_ids = batch.column(0).cast(pa.int64()).to_pylist()
        titles = batch.column(1).to_pylist()
        descriptions = batch.column(2).to_pylist()
        channels = batch.column(3).to_pylist()
        parent_channels = batch.column(4).to_pylist()
        tags = batch.column(5).to_pylist()

        for doc_id, title, description, channel, parent_channel, tag_list in zip(
            batch_ids,
            titles,
            descriptions,
            channels,
            parent_channels,
            tags,
            strict=True,
        ):
            doc_id = int(doc_id)
            if doc_id not in ids:
                continue
            out[doc_id] = {
                "title": title,
                "description": description,
                "channel": channel,
                "parent_channel": parent_channel,
                "tags": tags_to_text(tag_list),
            }
            if len(out) == len(ids):
                return out

    return out


def save_results(output_path: Path, rows: list[dict]) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("score", pa.float32()),
            pa.field("title", pa.string()),
            pa.field("description", pa.string()),
            pa.field("channel", pa.string()),
            pa.field("parent_channel", pa.string()),
            pa.field("tags", pa.string()),
        ]
    )
    batch = pa.record_batch(
        [
            pa.array([row["id"] for row in rows], type=pa.int64()),
            pa.array([row["score"] for row in rows], type=pa.float32()),
            pa.array([row["title"] for row in rows], type=pa.string()),
            pa.array([row["description"] for row in rows], type=pa.string()),
            pa.array([row["channel"] for row in rows], type=pa.string()),
            pa.array([row["parent_channel"] for row in rows], type=pa.string()),
            pa.array([row["tags"] for row in rows], type=pa.string()),
        ],
        schema=schema,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_batches([batch]), output_path.as_posix(), compression="zstd")


def main() -> None:
    args = parse_args()
    phrase, terms = normalize_query(args.query)
    print(
        json.dumps(
            {
                "query": args.query,
                "phrase": phrase,
                "term_mode": args.term_mode,
                "seed_limit": args.seed_limit,
                "top_k": args.top_k,
                "min_score": args.min_score,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    seed_ids, seed_meta, scanned = find_seed_ids(
        args.metadata,
        phrase,
        terms,
        args.term_mode,
        args.batch_size,
        args.seed_limit,
    )
    print(f"[seed] done scanned={scanned} matched={len(seed_ids)}", flush=True)
    if not seed_ids:
        raise ValueError("No seed documents matched the query")

    centroid, used = accumulate_centroid(args.tfidf, seed_ids, args.batch_size)
    print(f"[centroid] done used={used} dims={centroid.shape[0]}", flush=True)

    ranked = score_all(
        args.tfidf,
        centroid,
        args.batch_size,
        args.top_k,
        args.min_score,
        seed_ids,
        args.exclude_seeds,
    )
    print(f"[score] done hits={len(ranked)}", flush=True)

    result_ids = {doc_id for _, doc_id in ranked}
    meta = fetch_result_metadata(args.metadata, result_ids, args.batch_size)

    rows: list[dict] = []
    for score, doc_id in ranked:
        info = meta.get(doc_id, {})
        rows.append(
            {
                "id": doc_id,
                "score": np.float32(score).item(),
                "title": info.get("title"),
                "description": info.get("description"),
                "channel": info.get("channel"),
                "parent_channel": info.get("parent_channel"),
                "tags": info.get("tags"),
            }
        )

    if args.output:
        save_results(args.output, rows)
        print(f"[output] {args.output}", flush=True)

    for index, row in enumerate(rows[: args.show], start=1):
        print(
            json.dumps(
                {
                    "rank": index,
                    "id": row["id"],
                    "score": row["score"],
                    "channel": row["channel"],
                    "title": row["title"],
                    "tags": row["tags"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
