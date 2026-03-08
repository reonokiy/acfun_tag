#!/usr/bin/env python3
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DEFAULT_FLATTENED = Path("data/acfun.videoinfo.20260307.full.flattened.parquet")
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
FLATTENED_COLUMNS = DISPLAY_COLUMNS + ["payload_createTimeMillis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score candidate videos from an id parquet by joining back to the flattened parquet."
    )
    parser.add_argument("ids_parquet", type=Path, help="Candidate id parquet, usually exported from BM25")
    parser.add_argument("--flattened", type=Path, default=DEFAULT_FLATTENED, help="Flattened parquet path")
    parser.add_argument("--output-parquet", type=Path, default=None, help="Optional scored parquet path")
    parser.add_argument(
        "--selected-output-parquet",
        type=Path,
        default=None,
        help="Optional selected top-k parquet path",
    )
    parser.add_argument(
        "--include-info-columns",
        action="store_true",
        help="Include title/user/time and engagement fields in output parquet files for manual inspection.",
    )
    parser.add_argument("--top-k", type=int, default=1000, help="Selected rows to output")
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=540.0,
        help="Half-life for time decay in days. Set a very large value to minimize decay.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("score", "time-quota"),
        default="time-quota",
        help="How to choose the selected top-k rows.",
    )
    parser.add_argument(
        "--time-bucket",
        choices=("year", "quarter", "month"),
        default="month",
        help="Bucket granularity used by time-quota selection.",
    )
    parser.add_argument(
        "--min-per-bucket",
        type=int,
        default=1,
        help="Minimum guaranteed selections for each non-empty time bucket in time-quota mode.",
    )
    parser.add_argument("--weight-view", type=float, default=DEFAULT_METRICS["payload_viewCount"])
    parser.add_argument("--weight-like", type=float, default=DEFAULT_METRICS["payload_likeCount"])
    parser.add_argument("--weight-comment", type=float, default=DEFAULT_METRICS["payload_commentCountRealValue"])
    parser.add_argument("--weight-banana", type=float, default=DEFAULT_METRICS["payload_bananaCount"])
    parser.add_argument("--weight-danmaku", type=float, default=DEFAULT_METRICS["payload_danmakuCount"])
    parser.add_argument("--weight-stow", type=float, default=DEFAULT_METRICS["payload_stowCount"])
    parser.add_argument("--weight-share", type=float, default=DEFAULT_METRICS["payload_shareCount"])
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
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


def load_candidates(ids_parquet: Path, flattened: Path) -> pd.DataFrame:
    ids_df = pd.read_parquet(ids_parquet)
    if "id" not in ids_df.columns:
        raise ValueError(f"id parquet {ids_parquet} does not contain an 'id' column")
    ids_df = ids_df.drop_duplicates(subset=["id"]).copy()

    con = duckdb.connect()
    try:
        con.register("candidate_ids", ids_df)
        flat_cols_sql = ", ".join(f"f.{col}" for col in FLATTENED_COLUMNS)
        sql = f"""
            SELECT
                c.*,
                {flat_cols_sql}
            FROM candidate_ids AS c
            JOIN read_parquet(?) AS f
              ON c.id = f.id
        """
        joined = con.execute(sql, [flattened.as_posix()]).fetchdf()
    finally:
        con.close()
    return joined


def normalize_metric(series: pd.Series) -> tuple[pd.Series, float]:
    logged = series.fillna(0).clip(lower=0).map(lambda value: np.log1p(value))
    upper = float(logged.quantile(0.99))
    if upper <= 0 or logged.nunique() <= 1:
        return pd.Series(0.0, index=series.index), upper
    return logged.rank(method="average", pct=True), upper


def score_candidates(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
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
    guaranteed = {bucket: min(min_per_bucket, int(count)) for bucket, count in counts.items()}
    guaranteed_total = sum(guaranteed.values())
    if guaranteed_total > top_k:
        raise ValueError(f"top_k={top_k} is too small for min_per_bucket={min_per_bucket}")

    remaining_counts = counts - pd.Series(guaranteed)
    remaining_slots = top_k - guaranteed_total
    if remaining_slots <= 0:
        return guaranteed
    eligible = remaining_counts[remaining_counts > 0]
    if eligible.empty:
        return guaranteed

    expected = (eligible / eligible.sum()) * remaining_slots
    base = expected.astype(int)
    quotas = guaranteed.copy()
    for bucket, value in base.items():
        quotas[bucket] = quotas.get(bucket, 0) + int(value)
    remaining = remaining_slots - int(base.sum())
    remainders = (expected - base).sort_values(ascending=False)
    for bucket in remainders.index[:remaining]:
        quotas[bucket] = quotas.get(bucket, 0) + 1
    return quotas


def select_top_k(df: pd.DataFrame, top_k: int, selection_mode: str, time_bucket: str, min_per_bucket: int) -> pd.DataFrame:
    if selection_mode == "score":
        return df.sort_values("score", ascending=False).head(top_k).copy()

    bucketed = add_time_bucket(df, time_bucket)
    counts = bucketed["time_bucket"].value_counts().sort_index()
    quotas = allocate_bucket_quotas(counts, top_k, min_per_bucket)
    selected_parts: list[pd.DataFrame] = []
    for bucket, quota in quotas.items():
        part = bucketed[bucketed["time_bucket"] == bucket].sort_values("score", ascending=False).head(quota)
        if not part.empty:
            selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else bucketed.head(0).copy()
    if len(selected) < top_k:
        extra = bucketed[~bucketed["id"].isin(selected["id"])].sort_values("score", ascending=False).head(top_k - len(selected))
        selected = pd.concat([selected, extra], ignore_index=True)
    return selected.head(top_k).copy()


def summarize_bucket_distribution(df: pd.DataFrame) -> dict[str, float]:
    shares = df["time_bucket"].value_counts(normalize=True).sort_index()
    return {str(bucket): float(share) for bucket, share in shares.items()}


def build_output_frame(df: pd.DataFrame, include_info_columns: bool) -> pd.DataFrame:
    base_columns = [
        "id",
        "score_before_decay",
        "time_decay",
        "score",
        "age_days",
        "time_bucket",
    ]
    passthrough_columns = [
        "rank",
        "query_text",
        "query_terms",
        "total_score",
        "title_score",
        "description_score",
        "tags_score",
        "parent_score",
        "channel_score",
    ]
    columns = [col for col in base_columns + passthrough_columns if col in df.columns]
    if include_info_columns:
        columns += [col for col in DISPLAY_COLUMNS if col in df.columns and col not in columns]
    return df[columns].copy()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    weights = build_weight_map(args)

    candidates = load_candidates(args.ids_parquet, args.flattened)
    ranked, clip_points = score_candidates(candidates, weights)
    ranked = apply_time_decay(ranked, args.half_life_days)
    ranked = add_time_bucket(ranked, args.time_bucket)
    selected = select_top_k(ranked, args.top_k, args.selection_mode, args.time_bucket, args.min_per_bucket)

    ranked_output = build_output_frame(ranked, args.include_info_columns)
    selected_output = build_output_frame(selected, args.include_info_columns)

    if args.output_parquet is not None:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        ranked_output.to_parquet(args.output_parquet, index=False)
    if args.selected_output_parquet is not None:
        args.selected_output_parquet.parent.mkdir(parents=True, exist_ok=True)
        selected_output.to_parquet(args.selected_output_parquet, index=False)

    preview_columns = [col for col in DISPLAY_COLUMNS if col in selected.columns] + ["time_bucket", "score_before_decay", "time_decay", "score"]
    print(selected.sort_values("score", ascending=False).head(min(20, len(selected)))[preview_columns].to_json(orient="records", force_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "ids_parquet": args.ids_parquet.as_posix(),
                "flattened": args.flattened.as_posix(),
                "candidate_rows": int(len(ranked)),
                "selected_rows": int(len(selected)),
                "output_parquet": None if args.output_parquet is None else args.output_parquet.as_posix(),
                "selected_output_parquet": None if args.selected_output_parquet is None else args.selected_output_parquet.as_posix(),
                "include_info_columns": args.include_info_columns,
                "selection_mode": args.selection_mode,
                "time_bucket": args.time_bucket,
                "min_per_bucket": args.min_per_bucket,
                "half_life_days": args.half_life_days,
                "source_time_distribution": summarize_bucket_distribution(ranked),
                "selected_time_distribution": summarize_bucket_distribution(selected),
                "weights": weights,
                "log_clip_p99": clip_points,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
