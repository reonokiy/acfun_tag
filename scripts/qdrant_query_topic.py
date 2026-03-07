#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
from qdrant_client import QdrantClient, models


DEFAULT_COLLECTION = "acfun_tfidf"
SPARSE_VECTOR_NAME = "tfidf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query topic neighbors from Qdrant sparse TF-IDF vectors."
    )
    parser.add_argument("query", help="Keyword or phrase used to bootstrap the topic")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/acfun.videoinfo.20260307.full.flattened.parquet"),
        help="Metadata parquet path used to find seed documents",
    )
    parser.add_argument("--url", default="http://127.0.0.1:6333", help="Qdrant URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Rows per metadata scan batch")
    parser.add_argument("--seed-limit", type=int, default=5_000, help="Maximum keyword seed documents")
    parser.add_argument("--top-k", type=int, default=100, help="Number of Qdrant hits to request")
    parser.add_argument("--min-score", type=float, default=0.15, help="Minimum cosine similarity")
    parser.add_argument("--retrieve-batch-size", type=int, default=2_000, help="Seed vector retrieval batch size")
    parser.add_argument("--timeout", type=int, default=120, help="Qdrant request timeout in seconds")
    parser.add_argument(
        "--seed-fields",
        default="title,description,channel,parent_channel,tags",
        help="Comma-separated fields used to match query when selecting seed documents",
    )
    parser.add_argument(
        "--term-mode",
        choices=("phrase", "all", "any"),
        default="phrase",
        help="How to match the query against metadata text",
    )
    parser.add_argument(
        "--exclude-seeds",
        action="store_true",
        help="Exclude seed documents from final results",
    )
    parser.add_argument("--show", type=int, default=20, help="How many hits to print")
    return parser.parse_args()


def normalize_query(text: str) -> tuple[str, list[str]]:
    phrase = " ".join(text.strip().lower().split())
    terms = [term for term in phrase.split(" ") if term]
    return phrase, terms


def tags_to_text(tag_value) -> str:
    if not tag_value:
        return ""
    names = []
    for item in tag_value:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = None
        if name:
            names.append(str(name))
    return " ".join(names)


def build_haystack(title, description, channel, parent_channel, tag_list) -> str:
    return " \n ".join(
        [
            title or "",
            description or "",
            channel or "",
            parent_channel or "",
            tags_to_text(tag_list),
        ]
    ).lower()


def build_field_texts(title, description, channel, parent_channel, tag_list) -> dict[str, str]:
    return {
        "title": (title or "").lower(),
        "description": (description or "").lower(),
        "channel": (channel or "").lower(),
        "parent_channel": (parent_channel or "").lower(),
        "tags": tags_to_text(tag_list).lower(),
    }


def row_matches(text: str, phrase: str, terms: list[str], mode: str) -> bool:
    if mode == "phrase":
        return phrase in text
    if mode == "all":
        return all(term in text for term in terms)
    return any(term in text for term in terms)


def find_seed_ids(
    metadata_path: Path,
    phrase: str,
    terms: list[str],
    mode: str,
    batch_size: int,
    seed_limit: int,
    seed_fields: set[str],
) -> list[int]:
    dataset = ds.dataset(metadata_path.as_posix(), format="parquet")
    scanner = dataset.scanner(
        columns=[
            "id",
            "payload_title",
            "payload_description",
            "payload_channel_name",
            "payload_channel_parentName",
            "payload_tagList",
        ],
        batch_size=batch_size,
        use_threads=True,
    )

    seed_ids: list[int] = []
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
            field_texts = build_field_texts(title, description, channel, parent_channel, tag_list)
            haystack = " \n ".join(
                field_texts[name]
                for name in ("title", "description", "channel", "parent_channel", "tags")
                if name in seed_fields
            )
            if row_matches(haystack, phrase, terms, mode):
                seed_ids.append(int(doc_id))
                if len(seed_ids) >= seed_limit:
                    return seed_ids
        if (batch_index + 1) % 20 == 0:
            print(f"[seed] batches={batch_index + 1} matched={len(seed_ids)}", flush=True)

    return seed_ids


def build_centroid(
    client: QdrantClient,
    collection: str,
    seed_ids: list[int],
    retrieve_batch_size: int,
    timeout: int,
) -> models.SparseVector:
    accum = defaultdict(float)
    used = 0
    for start in range(0, len(seed_ids), retrieve_batch_size):
        batch_ids = seed_ids[start : start + retrieve_batch_size]
        records = client.retrieve(
            collection_name=collection,
            ids=batch_ids,
            with_payload=False,
            with_vectors=[SPARSE_VECTOR_NAME],
            timeout=timeout,
        )
        for record in records:
            vector_map = record.vector or {}
            sparse = vector_map.get(SPARSE_VECTOR_NAME)
            if sparse is None:
                continue
            for index, value in zip(sparse.indices, sparse.values, strict=True):
                accum[int(index)] += float(value)
            used += 1
        if (start // retrieve_batch_size + 1) % 10 == 0:
            print(f"[centroid] retrieved={min(start + retrieve_batch_size, len(seed_ids))}", flush=True)

    if used == 0:
        raise ValueError("No usable seed vectors found in Qdrant")

    indices = sorted(accum.keys())
    values = [accum[index] / used for index in indices]
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0.0:
        raise ValueError("Centroid vector is zero")
    values = [value / norm for value in values]
    print(f"[centroid] used={used} nnz={len(indices)}", flush=True)
    return models.SparseVector(indices=indices, values=values)


def main() -> None:
    args = parse_args()
    phrase, terms = normalize_query(args.query)
    seed_fields = {field.strip() for field in args.seed_fields.split(",") if field.strip()}
    valid_fields = {"title", "description", "channel", "parent_channel", "tags"}
    invalid_fields = seed_fields - valid_fields
    if invalid_fields:
        raise ValueError(f"Unsupported seed fields: {sorted(invalid_fields)}")
    if not seed_fields:
        raise ValueError("seed-fields must not be empty")
    print(
        json.dumps(
            {
                "query": args.query,
                "phrase": phrase,
                "seed_limit": args.seed_limit,
                "top_k": args.top_k,
                "min_score": args.min_score,
                "term_mode": args.term_mode,
                "seed_fields": sorted(seed_fields),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    seed_ids = find_seed_ids(
        args.metadata,
        phrase,
        terms,
        args.term_mode,
        args.batch_size,
        args.seed_limit,
        seed_fields,
    )
    print(f"[seed] done matched={len(seed_ids)}", flush=True)
    if not seed_ids:
        raise ValueError("No seed documents matched the query")

    client = QdrantClient(url=args.url, timeout=args.timeout)
    centroid = build_centroid(client, args.collection, seed_ids, args.retrieve_batch_size, args.timeout)
    response = client.query_points(
        collection_name=args.collection,
        query=centroid,
        using=SPARSE_VECTOR_NAME,
        limit=args.top_k + (len(seed_ids) if args.exclude_seeds else 0),
        with_payload=True,
        with_vectors=False,
        score_threshold=args.min_score,
        timeout=args.timeout,
    )

    emitted = 0
    seed_set = set(seed_ids)
    for point in response.points:
        if args.exclude_seeds and int(point.id) in seed_set:
            continue
        payload = point.payload or {}
        emitted += 1
        print(
            json.dumps(
                {
                    "rank": emitted,
                    "id": int(point.id),
                    "score": float(point.score),
                    "channel": payload.get("channel"),
                    "title": payload.get("title"),
                    "tags": payload.get("tags"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if emitted >= args.show:
            break


if __name__ == "__main__":
    main()
