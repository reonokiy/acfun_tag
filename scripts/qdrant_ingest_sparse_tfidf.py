#!/usr/bin/env python3
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
from qdrant_client import QdrantClient, models


DEFAULT_COLLECTION = "acfun_tfidf"
SPARSE_VECTOR_NAME = "tfidf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest sparse TF-IDF parquet plus metadata parquet into Qdrant."
    )
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
    parser.add_argument("--url", default="http://127.0.0.1:6333", help="Qdrant URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=2_000, help="Rows per upsert batch")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingest",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ingest if the collection already exists",
    )
    return parser.parse_args()


def build_payload(title, description, channel, parent_channel, tags) -> dict:
    tag_names = []
    if tags:
        for item in tags:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = None
            if name:
                tag_names.append(str(name))
    return {
        "title": title,
        "description": description,
        "channel": channel,
        "parent_channel": parent_channel,
        "tags": tag_names,
        "search_text": "\n".join(
            [
                title or "",
                description or "",
                channel or "",
                parent_channel or "",
                " ".join(tag_names),
            ]
        ),
    }


def ensure_collection(client: QdrantClient, collection: str, recreate: bool, skip_existing: bool) -> bool:
    exists = client.collection_exists(collection)
    if exists and skip_existing:
        print(f"[collection] skip existing {collection}", flush=True)
        return False
    if exists and recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config={},
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
            on_disk_payload=True,
        )
        print(f"[collection] recreated {collection}", flush=True)
        return True
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={},
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
            on_disk_payload=True,
        )
        print(f"[collection] created {collection}", flush=True)
        return True
    print(f"[collection] using existing {collection}", flush=True)
    return True


def iter_joined_batches(metadata_path: Path, tfidf_path: Path, batch_size: int):
    meta_dataset = ds.dataset(metadata_path.as_posix(), format="parquet")
    tfidf_dataset = ds.dataset(tfidf_path.as_posix(), format="parquet")
    meta_scanner = meta_dataset.scanner(
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
    tfidf_scanner = tfidf_dataset.scanner(
        columns=["id", "vector"],
        batch_size=batch_size,
        use_threads=True,
    )

    for batch_index, (meta_batch, vec_batch) in enumerate(
        zip(meta_scanner.to_batches(), tfidf_scanner.to_batches(), strict=True)
    ):
        meta_ids = meta_batch.column(0).cast(pa.int64()).to_pylist()
        vec_ids = vec_batch.column(0).cast(pa.int64()).to_pylist()
        if meta_ids != vec_ids:
            raise ValueError(f"Metadata / TF-IDF id mismatch in batch {batch_index}")
        yield batch_index, meta_batch, vec_batch


def ingest(args: argparse.Namespace) -> None:
    client = QdrantClient(url=args.url)
    should_ingest = ensure_collection(client, args.collection, args.recreate, args.skip_existing)
    if not should_ingest:
        return

    total_rows = 0
    for batch_index, meta_batch, vec_batch in iter_joined_batches(args.metadata, args.tfidf, args.batch_size):
        ids = meta_batch.column(0).cast(pa.int64()).to_pylist()
        titles = meta_batch.column(1).to_pylist()
        descriptions = meta_batch.column(2).to_pylist()
        channels = meta_batch.column(3).to_pylist()
        parent_channels = meta_batch.column(4).to_pylist()
        tags = meta_batch.column(5).to_pylist()

        vector_col = vec_batch.column(1)
        indices_rows = vector_col.field("indices").to_pylist()
        values_rows = vector_col.field("values").to_pylist()

        points = []
        for doc_id, title, description, channel, parent_channel, tag_list, indices, values in zip(
            ids,
            titles,
            descriptions,
            channels,
            parent_channels,
            tags,
            indices_rows,
            values_rows,
            strict=True,
        ):
            points.append(
                models.PointStruct(
                    id=int(doc_id),
                    vector={
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=indices or [],
                            values=values or [],
                        )
                    },
                    payload=build_payload(title, description, channel, parent_channel, tag_list),
                )
            )

        client.upsert(collection_name=args.collection, points=points, wait=True)
        total_rows += len(points)
        if (batch_index + 1) % 10 == 0:
            print(f"[ingest] batches={batch_index + 1} rows={total_rows}", flush=True)

    print(f"[done] collection={args.collection} rows={total_rows}", flush=True)


def main() -> None:
    args = parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
