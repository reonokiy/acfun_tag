#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce a sparse TF-IDF parquet (id + vector{indices,values}) with "
            "TruncatedSVD and write a dense parquet."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/acfun.videoinfo.20260307.full.tfidf.parquet"),
        help="Input sparse TF-IDF parquet path",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("data/acfun.videoinfo.20260307.full.tfidf.svd1024.parquet"),
        help="Output dense parquet path",
    )
    parser.add_argument("--id-column", default="id", help="ID column name")
    parser.add_argument("--vector-column", default="vector", help="Sparse vector column name")
    parser.add_argument("--batch-size", type=int, default=20_000, help="Rows per transform batch")
    parser.add_argument(
        "--fit-rows",
        type=int,
        default=200_000,
        help="Approximate number of rows sampled to fit SVD",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=1024,
        help="TruncatedSVD output dimensions",
    )
    parser.add_argument("--n-iter", type=int, default=7, help="TruncatedSVD n_iter")
    parser.add_argument(
        "--oversamples",
        type=int,
        default=32,
        help="TruncatedSVD n_oversamples",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--features",
        type=int,
        default=None,
        help="Input feature count. Default: infer from max sparse index + 1",
    )
    return parser.parse_args()


def count_rows_and_features(
    dataset: ds.Dataset,
    vector_column: str,
    batch_size: int,
    explicit_features: int | None,
) -> tuple[int, int]:
    total_rows = 0
    max_index = -1
    scanner = dataset.scanner(columns=[vector_column], batch_size=batch_size, use_threads=True)

    for batch_index, batch in enumerate(scanner.to_batches()):
        total_rows += batch.num_rows
        if explicit_features is None:
            index_values = batch.column(0).field("indices").values
            if len(index_values):
                batch_max = int(index_values.cast(pa.int64()).to_numpy(zero_copy_only=False).max())
                if batch_max > max_index:
                    max_index = batch_max
        if (batch_index + 1) % 20 == 0:
            print(f"[scan] batches={batch_index + 1} rows={total_rows}", flush=True)

    num_features = explicit_features if explicit_features is not None else max_index + 1
    if num_features <= 0:
        raise ValueError("Failed to infer feature count from sparse vectors")
    return total_rows, num_features


def batch_to_csr(batch: pa.RecordBatch, id_column: str, vector_column: str, num_features: int):
    ids = batch.column(batch.schema.get_field_index(id_column)).cast(pa.int64()).to_numpy(zero_copy_only=False)
    vector_array = batch.column(batch.schema.get_field_index(vector_column))
    indices_array = vector_array.field("indices")
    values_array = vector_array.field("values")

    indptr = indices_array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    indptr = indptr - indptr[0]
    indices = indices_array.values.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    values = values_array.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    matrix = sparse.csr_matrix((values, indices, indptr), shape=(batch.num_rows, num_features), dtype=np.float32)
    return ids, matrix


def sample_fit_matrix(
    dataset: ds.Dataset,
    args: argparse.Namespace,
    total_rows: int,
    num_features: int,
) -> sparse.csr_matrix:
    rng = np.random.default_rng(args.random_state)
    sample_rate = min(1.0, args.fit_rows / max(total_rows, 1))
    sampled_parts: list[sparse.csr_matrix] = []
    sampled_rows = 0

    scanner = dataset.scanner(
        columns=[args.id_column, args.vector_column],
        batch_size=args.batch_size,
        use_threads=True,
    )
    for batch_index, batch in enumerate(scanner.to_batches()):
        _, matrix = batch_to_csr(batch, args.id_column, args.vector_column, num_features)
        if sample_rate >= 1.0:
            sampled = matrix
        else:
            keep_mask = rng.random(matrix.shape[0]) < sample_rate
            if not keep_mask.any():
                continue
            sampled = matrix[keep_mask]
        if sampled.shape[0] == 0:
            continue

        sampled_parts.append(sampled)
        sampled_rows += sampled.shape[0]
        if (batch_index + 1) % 20 == 0:
            print(f"[sample] batches={batch_index + 1} sampled_rows={sampled_rows}", flush=True)

    if not sampled_parts:
        raise ValueError("No rows sampled for SVD fitting")

    fit_matrix = sparse.vstack(sampled_parts, format="csr", dtype=np.float32)
    if fit_matrix.shape[0] > args.fit_rows:
        choice = rng.choice(fit_matrix.shape[0], size=args.fit_rows, replace=False)
        choice.sort()
        fit_matrix = fit_matrix[choice]
    return fit_matrix


def fit_svd(fit_matrix: sparse.csr_matrix, args: argparse.Namespace) -> TruncatedSVD:
    if fit_matrix.shape[0] < args.components:
        raise ValueError(
            f"sampled rows ({fit_matrix.shape[0]}) must be >= components ({args.components})"
        )
    svd = TruncatedSVD(
        n_components=args.components,
        algorithm="randomized",
        n_iter=args.n_iter,
        n_oversamples=args.oversamples,
        random_state=args.random_state,
    )
    svd.fit(fit_matrix)
    explained = float(np.sum(svd.explained_variance_ratio_))
    print(f"[fit] rows={fit_matrix.shape[0]} features={fit_matrix.shape[1]} explained={explained:.6f}", flush=True)
    return svd


def build_output_schema(components: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), components)),
        ]
    )


def reduced_to_array(reduced: np.ndarray, components: int) -> pa.FixedSizeListArray:
    values = pa.array(np.asarray(reduced, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, list_size=components)


def write_reduced_parquet(
    dataset: ds.Dataset,
    args: argparse.Namespace,
    svd: TruncatedSVD,
    num_features: int,
) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = build_output_schema(args.components)
    writer = pq.ParquetWriter(args.output.as_posix(), schema=schema, compression="zstd")
    written = 0

    try:
        scanner = dataset.scanner(
            columns=[args.id_column, args.vector_column],
            batch_size=args.batch_size,
            use_threads=True,
        )
        for batch_index, batch in enumerate(scanner.to_batches()):
            ids, matrix = batch_to_csr(batch, args.id_column, args.vector_column, num_features)
            reduced = svd.transform(matrix).astype(np.float32, copy=False)
            output_batch = pa.record_batch(
                [
                    pa.array(ids, type=pa.int64()),
                    reduced_to_array(reduced, args.components),
                ],
                schema=schema,
            )
            writer.write_batch(output_batch)
            written += batch.num_rows
            if (batch_index + 1) % 20 == 0:
                print(f"[write] batches={batch_index + 1} rows={written}", flush=True)
    finally:
        writer.close()

    return written


def main() -> None:
    args = parse_args()
    dataset = ds.dataset(args.input.as_posix(), format="parquet")
    field_names = set(dataset.schema.names)
    if args.id_column not in field_names:
        raise ValueError(f"id column {args.id_column!r} not found")
    if args.vector_column not in field_names:
        raise ValueError(f"vector column {args.vector_column!r} not found")

    print(f"input={args.input} output={args.output}", flush=True)
    total_rows, num_features = count_rows_and_features(dataset, args.vector_column, args.batch_size, args.features)
    print(f"[scan] done rows={total_rows} features={num_features}", flush=True)

    fit_matrix = sample_fit_matrix(dataset, args, total_rows, num_features)
    print(f"[sample] done rows={fit_matrix.shape[0]}", flush=True)

    svd = fit_svd(fit_matrix, args)
    written = write_reduced_parquet(dataset, args, svd, num_features)
    print(f"[write] done rows={written}", flush=True)


if __name__ == "__main__":
    main()
