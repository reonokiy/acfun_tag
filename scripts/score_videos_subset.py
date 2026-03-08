#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


DEFAULT_INPUT = Path("data/acfun.videoinfo.20260307.full.flattened.parquet")
DEFAULT_OUTPUT = Path("data/scores/video_score_subset.csv")
DEFAULT_METRICS = {
    "payload_viewCount": 0.05,
    "payload_likeCount": 0.25,
    "payload_commentCountRealValue": 0.15,
    "payload_bananaCount": 0.20,
    "payload_danmakuCount": 0.10,
    "payload_stowCount": 0.20,
    "payload_shareCount": 0.05,
}
DISPLAY_COLUMNS = [
    "id",
    "payload_title",
    "payload_user_name",
    "payload_createTime",
    "payload_viewCount",
    "payload_likeCount",
    "payload_commentCountRealValue",
    "payload_bananaCount",
    "payload_danmakuCount",
    "payload_stowCount",
    "payload_shareCount",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a small subset of AcFun videos with adjustable engagement weights."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input parquet path")
    parser.add_argument("--limit-rows", type=int, default=20000, help="Rows to load for the smoke test")
    parser.add_argument("--top-k", type=int, default=30, help="Top scored rows to print")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--sample-method",
        choices=("hash", "head"),
        default="hash",
        help="Subset selection method. 'hash' is a stable pseudo-random sample.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=540.0,
        help="Half-life for time decay in days",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("score", "time-quota"),
        default="time-quota",
        help="How to choose the displayed top-k results.",
    )
    parser.add_argument(
        "--time-bucket",
        choices=("year", "quarter", "month"),
        default="year",
        help="Bucket granularity used by time-quota selection.",
    )
    parser.add_argument(
        "--min-per-bucket",
        type=int,
        default=0,
        help="Minimum guaranteed selections for each non-empty time bucket in time-quota mode.",
    )
    parser.add_argument("--weight-view", type=float, default=DEFAULT_METRICS["payload_viewCount"], help="Weight for views")
    parser.add_argument("--weight-like", type=float, default=DEFAULT_METRICS["payload_likeCount"], help="Weight for likes")
    parser.add_argument(
        "--weight-comment",
        type=float,
        default=DEFAULT_METRICS["payload_commentCountRealValue"],
        help="Weight for comments",
    )
    parser.add_argument(
        "--weight-banana",
        type=float,
        default=DEFAULT_METRICS["payload_bananaCount"],
        help="Weight for banana count",
    )
    parser.add_argument(
        "--weight-danmaku",
        type=float,
        default=DEFAULT_METRICS["payload_danmakuCount"],
        help="Weight for danmaku count",
    )
    parser.add_argument("--weight-stow", type=float, default=DEFAULT_METRICS["payload_stowCount"], help="Weight for favorites")
    parser.add_argument("--weight-share", type=float, default=DEFAULT_METRICS["payload_shareCount"], help="Weight for shares")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level",
    )
    return parser.parse_args()


def build_weight_map(args: argparse.Namespace) -> dict[str, float]:
    return {
        "payload_viewCount": args.weight_view,
        "payload_likeCount": args.weight_like,
        "payload_commentCountRealValue": args.weight_comment,
        "payload_bananaCount": args.weight_banana,
        "payload_danmakuCount": args.weight_danmaku,
        "payload_stowCount": args.weight_stow,
        "payload_shareCount": args.weight_share,
    }


def load_subset(path: Path, limit_rows: int, sample_method: str) -> pd.DataFrame:
    columns = DISPLAY_COLUMNS.copy() + ["payload_createTimeMillis"]
    if sample_method == "head":
        dataset = ds.dataset(path.as_posix(), format="parquet")
        return dataset.head(limit_rows, columns=columns).to_pandas()

    column_sql = ", ".join(columns)
    con = duckdb.connect()
    sql = f"""
        SELECT {column_sql}
        FROM read_parquet(?)
        WHERE id IS NOT NULL
        ORDER BY hash(id)
        LIMIT ?
    """
    try:
        return con.execute(sql, [path.as_posix(), limit_rows]).fetchdf()
    finally:
        con.close()


def normalize_metric(series: pd.Series) -> tuple[pd.Series, float]:
    logged = series.fillna(0).clip(lower=0).map(lambda value: np.log1p(value))
    upper = float(logged.quantile(0.99))
    if upper <= 0 or logged.nunique() <= 1:
        return pd.Series(0.0, index=series.index), upper
    normalized = logged.rank(method="average", pct=True)
    return normalized, upper


def score_subset(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = df.copy()
    clip_points: dict[str, float] = {}
    weighted_score = pd.Series(0.0, index=scored.index)

    total_weight = sum(max(weight, 0.0) for weight in weights.values())
    if total_weight <= 0:
        raise ValueError("At least one positive weight is required")

    for metric, weight in weights.items():
        normalized, upper = normalize_metric(scored[metric])
        scored[f"{metric}_norm"] = normalized
        clip_points[metric] = upper
        weighted_score = weighted_score + normalized * max(weight, 0.0)

    scored["score"] = weighted_score / total_weight
    return scored.sort_values("score", ascending=False), clip_points


def apply_time_decay(df: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    scored = df.copy()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    age_days = ((now_ms - scored["payload_createTimeMillis"].fillna(now_ms)) / 86_400_000).clip(lower=0)
    time_decay = np.power(0.5, age_days / half_life_days)
    scored["age_days"] = age_days
    scored["time_decay"] = time_decay
    scored["score_before_decay"] = scored["score"]
    scored["score"] = scored["score_before_decay"] * scored["time_decay"]
    return scored.sort_values("score", ascending=False)


def add_time_bucket(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    scored = df.copy()
    timestamps = pd.to_datetime(scored["payload_createTimeMillis"], unit="ms", utc=True)
    if granularity == "year":
        bucket = timestamps.dt.strftime("%Y")
    elif granularity == "quarter":
        bucket = timestamps.dt.strftime("%Y") + "-Q" + timestamps.dt.quarter.astype(str)
    else:
        bucket = timestamps.dt.strftime("%Y-%m")
    scored["time_bucket"] = bucket.fillna("unknown")
    return scored


def allocate_bucket_quotas(counts: pd.Series, top_k: int, min_per_bucket: int) -> dict[str, int]:
    if min_per_bucket < 0:
        raise ValueError("min_per_bucket must be non-negative")
    if len(counts) == 0:
        return {}

    guaranteed = {bucket: min(min_per_bucket, int(count)) for bucket, count in counts.items()}
    guaranteed_total = sum(guaranteed.values())
    if guaranteed_total > top_k:
        raise ValueError(
            f"top_k={top_k} is too small for min_per_bucket={min_per_bucket} across {len(counts)} buckets"
        )

    remaining_counts = counts - pd.Series(guaranteed)
    remaining_slots = top_k - guaranteed_total
    if remaining_slots <= 0:
        return guaranteed

    eligible = remaining_counts[remaining_counts > 0]
    if eligible.empty:
        return guaranteed

    shares = eligible / eligible.sum()
    expected = shares * remaining_slots
    base = expected.astype(int)
    quotas = guaranteed.copy()
    for bucket, value in base.items():
        quotas[bucket] = quotas.get(bucket, 0) + int(value)

    remaining = remaining_slots - int(base.sum())
    remainders = (expected - base).sort_values(ascending=False)
    for bucket in remainders.index[:remaining]:
        quotas[bucket] = quotas.get(bucket, 0) + 1
    return quotas


def select_top_k(
    df: pd.DataFrame,
    top_k: int,
    selection_mode: str,
    time_bucket: str,
    min_per_bucket: int,
) -> pd.DataFrame:
    if selection_mode == "score":
        return df.sort_values("score", ascending=False).head(top_k).copy()

    bucketed = add_time_bucket(df, time_bucket)
    counts = bucketed["time_bucket"].value_counts().sort_index()
    quotas = allocate_bucket_quotas(counts, top_k, min_per_bucket)

    selected_parts: list[pd.DataFrame] = []
    for bucket, quota in quotas.items():
        if quota <= 0:
            continue
        part = (
            bucketed[bucketed["time_bucket"] == bucket]
            .sort_values("score", ascending=False)
            .head(quota)
        )
        if not part.empty:
            selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else bucketed.head(0).copy()
    selected = selected.sort_values(["time_bucket", "score"], ascending=[True, False])

    if len(selected) < top_k:
        missing = top_k - len(selected)
        extra = bucketed[~bucketed["id"].isin(selected["id"])].sort_values("score", ascending=False).head(missing)
        selected = pd.concat([selected, extra], ignore_index=True)

    return selected.head(top_k).copy()


def summarize_bucket_distribution(df: pd.DataFrame, bucket_col: str) -> dict[str, float]:
    shares = df[bucket_col].value_counts(normalize=True).sort_index()
    return {str(bucket): float(share) for bucket, share in shares.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    weights = build_weight_map(args)
    subset = load_subset(args.input, args.limit_rows, args.sample_method)
    ranked, clip_points = score_subset(subset, weights)
    ranked = apply_time_decay(ranked, args.half_life_days)
    ranked = add_time_bucket(ranked, args.time_bucket)
    selected = select_top_k(
        ranked,
        args.top_k,
        args.selection_mode,
        args.time_bucket,
        args.min_per_bucket,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)

    preview = selected[
        DISPLAY_COLUMNS + ["time_bucket", "score_before_decay", "time_decay", "score", "age_days"]
    ]
    print(preview.to_json(orient="records", force_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "rows_scored": int(len(ranked)),
                "top_k": args.top_k,
                "output": args.output.as_posix(),
                "sample_method": args.sample_method,
                "half_life_days": args.half_life_days,
                "selection_mode": args.selection_mode,
                "time_bucket": args.time_bucket,
                "min_per_bucket": args.min_per_bucket,
                "source_time_distribution": summarize_bucket_distribution(ranked, "time_bucket"),
                "selected_time_distribution": summarize_bucket_distribution(selected, "time_bucket"),
                "weights": weights,
                "log_clip_p99": clip_points,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
