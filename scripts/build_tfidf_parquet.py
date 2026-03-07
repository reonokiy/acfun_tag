#!/usr/bin/env python3
import argparse
import math
import os
import re
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import jieba
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from janome.tokenizer import Tokenizer as JanomeTokenizer
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_TITLE_CANDIDATES = ("payload_title", "title")
DEFAULT_DESCRIPTION_CANDIDATES = ("payload_description", "description")

WORKER_VECTORIZER: HashingVectorizer | None = None
WORKER_IDF: np.ndarray | None = None
JANOME_TOKENIZER: JanomeTokenizer | None = None

LATIN_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[._+-][0-9a-z]+)*")
PUNCT_ONLY_RE = re.compile(r"^[^\w\u3400-\u9fff\u3040-\u30ff]+$")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
TOKEN_CHUNK_RE = re.compile(r"[0-9a-z]+(?:[._+-][0-9a-z]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")

VECTOR_FIELDS = [
    pa.field("indices", pa.list_(pa.int32())),
    pa.field("values", pa.list_(pa.float32())),
]
VECTOR_TYPE = pa.struct(VECTOR_FIELDS)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("vector", VECTOR_TYPE),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TF-IDF parquet from a source parquet using title and "
            "description. Output schema: id, vector."
        )
    )
    parser.add_argument("input", type=Path, help="Input parquet path")
    parser.add_argument("output", type=Path, help="Output parquet path")
    parser.add_argument("--id-column", default="id", help="ID column name in the input parquet")
    parser.add_argument("--title-column", default=None, help="Title column name. Defaults to auto-detect.")
    parser.add_argument(
        "--description-column",
        default=None,
        help="Description column name. Defaults to auto-detect.",
    )
    parser.add_argument("--batch-size", type=int, default=50_000, help="Rows per batch")
    parser.add_argument("--features", type=int, default=1 << 20, help="Hash feature space size")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) - 1)),
        help="Worker process count",
    )
    parser.add_argument(
        "--analyzer",
        choices=("char", "char_wb", "word", "cjk_word"),
        default="cjk_word",
        help="HashingVectorizer analyzer",
    )
    parser.add_argument("--min-n", type=int, default=2, help="Lower bound of ngram range")
    parser.add_argument("--max-n", type=int, default=4, help="Upper bound of ngram range")
    return parser.parse_args()


def resolve_column(field_names: set[str], explicit: str | None, candidates: tuple[str, ...], label: str) -> str:
    if explicit:
        if explicit not in field_names:
            raise ValueError(f"{label} column {explicit!r} not found")
        return explicit
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    raise ValueError(f"Could not auto-detect {label} column from {candidates}")


def build_vectorizer(args: argparse.Namespace) -> HashingVectorizer:
    analyzer = args.analyzer
    kwargs: dict[str, object] = {
        "n_features": args.features,
        "alternate_sign": False,
        "norm": None,
        "binary": False,
        "lowercase": True,
        "ngram_range": (args.min_n, args.max_n),
    }
    if analyzer == "cjk_word":
        kwargs["analyzer"] = "word"
        kwargs["tokenizer"] = tokenize_cjk_text
        kwargs["token_pattern"] = None
    else:
        kwargs["analyzer"] = analyzer
    return HashingVectorizer(
        **kwargs,
    )


def get_janome_tokenizer() -> JanomeTokenizer:
    global JANOME_TOKENIZER
    if JANOME_TOKENIZER is None:
        JANOME_TOKENIZER = JanomeTokenizer()
    return JANOME_TOKENIZER


def tokenize_japanese(text: str) -> list[str]:
    tokenizer = get_janome_tokenizer()
    tokens: list[str] = []
    for token in tokenizer.tokenize(text):
        surface = token.surface.strip().lower()
        if not surface or PUNCT_ONLY_RE.match(surface):
            continue
        if HAN_RE.search(surface) or KANA_RE.search(surface):
            tokens.append(surface)
            continue
        tokens.extend(LATIN_TOKEN_RE.findall(surface))
    for chunk in TOKEN_CHUNK_RE.findall(text.lower()):
        if HAN_RE.search(chunk) and len(chunk) > 1:
            tokens.append(chunk)
    return tokens


def tokenize_chinese(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in TOKEN_CHUNK_RE.findall(text.lower()):
        if HAN_RE.search(chunk):
            pieces = [piece.strip() for piece in jieba.lcut(chunk, HMM=True) if piece.strip()]
            tokens.extend(pieces or [chunk])
            continue
        tokens.append(chunk)
    return tokens


def tokenize_cjk_text(text: str) -> list[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []
    if KANA_RE.search(normalized):
        return tokenize_japanese(normalized)
    return tokenize_chinese(normalized)


def combine_text(title_array: pa.Array, description_array: pa.Array) -> list[str]:
    titles = title_array.to_pylist()
    descriptions = description_array.to_pylist()
    return [f"{title or ''}\n{description or ''}" for title, description in zip(titles, descriptions, strict=True)]


def iter_batches(
    dataset: ds.Dataset,
    id_column: str | None,
    title_col: str,
    description_col: str,
    batch_size: int,
):
    columns = [title_col, description_col] if id_column is None else [id_column, title_col, description_col]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    for batch_index, batch in enumerate(scanner.to_batches()):
        if id_column is None:
            ids = None
            title_arr = batch.column(0)
            desc_arr = batch.column(1)
        else:
            ids = batch.column(0).cast(pa.int64()).to_numpy(zero_copy_only=False)
            title_arr = batch.column(1)
            desc_arr = batch.column(2)
        yield batch_index, ids, combine_text(title_arr, desc_arr)


def init_pass1_worker(vectorizer: HashingVectorizer) -> None:
    global WORKER_VECTORIZER
    WORKER_VECTORIZER = vectorizer


def pass1_job(payload: tuple[int, list[str]]) -> tuple[int, int, np.ndarray]:
    global WORKER_VECTORIZER
    batch_index, texts = payload
    matrix = WORKER_VECTORIZER.transform(texts)
    doc_freq = matrix.getnnz(axis=0).astype(np.int32, copy=False)
    return batch_index, matrix.shape[0], doc_freq


def compute_idf(
    dataset: ds.Dataset,
    args: argparse.Namespace,
    title_col: str,
    description_col: str,
) -> tuple[np.ndarray, int]:
    vectorizer = build_vectorizer(args)
    doc_freq = np.zeros(args.features, dtype=np.int64)
    total_docs = 0

    if args.workers == 1:
        init_pass1_worker(vectorizer)
        for batch_index, _, texts in iter_batches(dataset, None, title_col, description_col, args.batch_size):
            _, rows, batch_df = pass1_job((batch_index, texts))
            doc_freq += batch_df
            total_docs += rows
            if (batch_index + 1) % 20 == 0:
                print(f"[pass1] batches={batch_index + 1} rows={total_docs}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_pass1_worker,
            initargs=(vectorizer,),
        ) as pool:
            pending = set()
            source = iter_batches(dataset, None, title_col, description_col, args.batch_size)
            max_pending = args.workers * 2

            while True:
                while len(pending) < max_pending:
                    try:
                        batch_index, _, texts = next(source)
                    except StopIteration:
                        break
                    pending.add(pool.submit(pass1_job, (batch_index, texts)))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    batch_index, rows, batch_df = future.result()
                    doc_freq += batch_df
                    total_docs += rows
                    if (batch_index + 1) % 20 == 0:
                        print(f"[pass1] batches={batch_index + 1} rows={total_docs}", flush=True)

    idf = np.log((1.0 + total_docs) / (1.0 + doc_freq)) + 1.0
    return idf.astype(np.float32, copy=False), total_docs


def init_pass2_worker(vectorizer: HashingVectorizer, idf: np.ndarray, shard_dir: str) -> None:
    global WORKER_VECTORIZER, WORKER_IDF
    WORKER_VECTORIZER = vectorizer
    WORKER_IDF = idf
    Path(shard_dir).mkdir(parents=True, exist_ok=True)


def pass2_job(payload: tuple[int, np.ndarray, list[str], str]) -> tuple[int, int, str]:
    global WORKER_VECTORIZER, WORKER_IDF
    batch_index, ids, texts, shard_dir = payload
    matrix = WORKER_VECTORIZER.transform(texts).tocsr()
    matrix.data *= WORKER_IDF[matrix.indices]

    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    row_norms[row_norms == 0.0] = 1.0
    row_ids = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    matrix.data /= row_norms[row_ids]

    indices_list: list[list[int]] = []
    values_list: list[list[float]] = []
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        indices_list.append(matrix.indices[start:end].astype(np.int32, copy=False).tolist())
        values_list.append(matrix.data[start:end].astype(np.float32, copy=False).tolist())

    vector_array = pa.StructArray.from_arrays(
        [
            pa.array(indices_list, type=pa.list_(pa.int32())),
            pa.array(values_list, type=pa.list_(pa.float32())),
        ],
        fields=VECTOR_FIELDS,
    )
    batch = pa.record_batch(
        [
            pa.array(ids, type=pa.int64()),
            vector_array,
        ],
        schema=OUTPUT_SCHEMA,
    )

    shard_path = Path(shard_dir) / f"part-{batch_index:06d}.parquet"
    writer = pq.ParquetWriter(shard_path.as_posix(), schema=OUTPUT_SCHEMA, compression="zstd")
    try:
        writer.write_batch(batch)
    finally:
        writer.close()
    return batch_index, len(ids), shard_path.as_posix()


def merge_shards(shard_dir: Path, output: Path) -> None:
    shard_paths = sorted(shard_dir.glob("part-*.parquet"))
    writer = pq.ParquetWriter(output.as_posix(), schema=OUTPUT_SCHEMA, compression="zstd")
    try:
        for shard_path in shard_paths:
            parquet_file = pq.ParquetFile(shard_path.as_posix())
            for batch in parquet_file.iter_batches():
                writer.write_batch(batch)
    finally:
        writer.close()


def write_tfidf(
    dataset: ds.Dataset,
    args: argparse.Namespace,
    id_column: str,
    title_col: str,
    description_col: str,
    idf: np.ndarray,
) -> int:
    vectorizer = build_vectorizer(args)
    shard_dir = args.output.parent / f".{args.output.name}.parts"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    if args.workers == 1:
        init_pass2_worker(vectorizer, idf, shard_dir.as_posix())
        for batch_index, ids, texts in iter_batches(dataset, id_column, title_col, description_col, args.batch_size):
            _, rows, _ = pass2_job((batch_index, ids, texts, shard_dir.as_posix()))
            total_rows += rows
            if (batch_index + 1) % 20 == 0:
                print(f"[pass2] batches={batch_index + 1} rows={total_rows}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_pass2_worker,
            initargs=(vectorizer, idf, shard_dir.as_posix()),
        ) as pool:
            pending = set()
            source = iter_batches(dataset, id_column, title_col, description_col, args.batch_size)
            max_pending = args.workers * 2

            while True:
                while len(pending) < max_pending:
                    try:
                        batch_index, ids, texts = next(source)
                    except StopIteration:
                        break
                    pending.add(pool.submit(pass2_job, (batch_index, ids, texts, shard_dir.as_posix())))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    batch_index, rows, _ = future.result()
                    total_rows += rows
                    if (batch_index + 1) % 20 == 0:
                        print(f"[pass2] batches={batch_index + 1} rows={total_rows}", flush=True)

    merge_shards(shard_dir, args.output)
    shutil.rmtree(shard_dir)
    return total_rows


def main() -> None:
    args = parse_args()
    dataset = ds.dataset(args.input.as_posix(), format="parquet")
    field_names = set(dataset.schema.names)

    if args.id_column not in field_names:
        raise ValueError(f"id column {args.id_column!r} not found")

    title_col = resolve_column(field_names, args.title_column, DEFAULT_TITLE_CANDIDATES, "title")
    description_col = resolve_column(
        field_names,
        args.description_column,
        DEFAULT_DESCRIPTION_CANDIDATES,
        "description",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"input={args.input} output={args.output} id={args.id_column} "
        f"title={title_col} description={description_col} "
        f"features={args.features} workers={args.workers}",
        flush=True,
    )
    idf, total_docs = compute_idf(dataset, args, title_col, description_col)
    print(f"[pass1] done rows={total_docs}", flush=True)
    written = write_tfidf(dataset, args, args.id_column, title_col, description_col, idf)
    print(f"[pass2] done rows={written}", flush=True)


if __name__ == "__main__":
    main()
