#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
from pathlib import Path

import duckdb
import jieba
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from janome.tokenizer import Tokenizer as JanomeTokenizer


DEFAULT_INPUT = Path("data/acfun.videoinfo.20260307.full.flattened.parquet")
DEFAULT_DB = Path("data/acfun.videoinfo.20260307.full.bm25.duckdb")
DEFAULT_TABLE = "videos"

DEFAULT_TITLE_CANDIDATES = ("payload_title", "title")
DEFAULT_DESCRIPTION_CANDIDATES = ("payload_description", "description")
DEFAULT_TAGS_CANDIDATES = ("payload_tagList", "tags")
DEFAULT_PARENT_CANDIDATES = ("payload_channel_parentName", "parent_channel")
DEFAULT_CHANNEL_CANDIDATES = ("payload_channel_name", "channel")

JANOME_TOKENIZER: JanomeTokenizer | None = None

LATIN_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[._+-][0-9a-z]+)*")
PUNCT_ONLY_RE = re.compile(r"^[^\w\u3400-\u9fff\u3040-\u30ff]+$")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
TOKEN_CHUNK_RE = re.compile(r"[0-9a-z]+(?:[._+-][0-9a-z]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")


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


def join_tokens(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(tokenize_cjk_text(text))


def tags_to_text(value) -> str:
    if not value:
        return ""
    pieces: list[str] = []
    for item in value:
        if not item:
            continue
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = getattr(item, "get", lambda *_: None)("name")
        if name:
            pieces.append(str(name))
    return " ".join(pieces)


def resolve_column(field_names: set[str], explicit: str | None, candidates: tuple[str, ...], label: str) -> str:
    if explicit:
        if explicit not in field_names:
            raise ValueError(f"{label} column {explicit!r} not found")
        return explicit
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    raise ValueError(f"Could not auto-detect {label} column from {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a DuckDB BM25 index from the flattened AcFun parquet."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the BM25 DuckDB index")
    build.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Flattened parquet path")
    build.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB output path")
    build.add_argument("--table", default=DEFAULT_TABLE, help="DuckDB table name")
    build.add_argument("--batch-size", type=int, default=20_000, help="Rows per batch")
    build.add_argument("--limit-rows", type=int, default=None, help="Only ingest first N rows")
    build.add_argument("--overwrite", action="store_true", help="Drop existing table/index if present")
    build.add_argument("--title-column", default=None, help="Title column name")
    build.add_argument("--description-column", default=None, help="Description column name")
    build.add_argument("--tags-column", default=None, help="Tag-list column name")
    build.add_argument("--parent-column", default=None, help="Parent partition column name")
    build.add_argument("--channel-column", default=None, help="Channel partition column name")

    query = subparsers.add_parser("query", help="Search the BM25 DuckDB index")
    query.add_argument("terms", nargs="+", help="Query terms or phrases")
    query.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB path")
    query.add_argument("--table", default=DEFAULT_TABLE, help="DuckDB table name")
    query.add_argument("--top-k", type=int, default=20, help="How many hits to return")
    query.add_argument("--conjunctive", action="store_true", help="Require all terms per field")
    query.add_argument("--tokenize-query", action="store_true", help="Tokenize query text with the same CJK tokenizer")
    query.add_argument("--title-weight", type=float, default=3.0, help="Weight for title score")
    query.add_argument("--description-weight", type=float, default=1.0, help="Weight for description score")
    query.add_argument("--tags-weight", type=float, default=4.0, help="Weight for tags score")
    query.add_argument("--parent-weight", type=float, default=1.5, help="Weight for parent partition score")
    query.add_argument("--channel-weight", type=float, default=2.0, help="Weight for channel partition score")
    query.add_argument("--json", action="store_true", help="Print JSON instead of a text table")
    query.add_argument("--show-description-chars", type=int, default=120, help="Description preview length")
    query.add_argument("--output-parquet", type=Path, default=None, help="Optional parquet path for query results")
    query.add_argument(
        "--include-info-columns",
        action="store_true",
        help="Include title/description/tags/channel info columns in the output parquet",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level",
    )
    return parser.parse_args()


def create_schema_sql(table: str, overwrite: bool) -> list[str]:
    statements: list[str] = []
    if overwrite:
        statements.append(f"DROP TABLE IF EXISTS {table}")
    statements.append(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id BIGINT,
            title VARCHAR,
            description VARCHAR,
            tags VARCHAR,
            parent_channel VARCHAR,
            channel VARCHAR,
            title_terms VARCHAR,
            description_terms VARCHAR,
            tags_terms VARCHAR,
            parent_terms VARCHAR,
            channel_terms VARCHAR
        )
        """
    )
    return statements


def build_batches(
    input_path: Path,
    batch_size: int,
    limit_rows: int | None,
    title_column: str,
    description_column: str,
    tags_column: str,
    parent_column: str,
    channel_column: str,
):
    dataset = ds.dataset(input_path.as_posix(), format="parquet")
    scanner = dataset.scanner(
        columns=["id", title_column, description_column, tags_column, parent_column, channel_column],
        batch_size=batch_size,
        use_threads=True,
    )
    emitted = 0
    for batch_index, batch in enumerate(scanner.to_batches()):
        ids = batch.column(0).cast(pa.int64()).to_pylist()
        titles = batch.column(1).to_pylist()
        descriptions = batch.column(2).to_pylist()
        tags = batch.column(3).to_pylist()
        parents = batch.column(4).to_pylist()
        channels = batch.column(5).to_pylist()

        out_rows: list[dict[str, object]] = []
        for id_value, title, description, tag_list, parent, channel in zip(
            ids, titles, descriptions, tags, parents, channels, strict=True
        ):
            if limit_rows is not None and emitted >= limit_rows:
                return
            title_text = (title or "").strip()
            description_text = (description or "").strip()
            tags_text = tags_to_text(tag_list)
            parent_text = (parent or "").strip()
            channel_text = (channel or "").strip()
            out_rows.append(
                {
                    "id": id_value,
                    "title": title_text,
                    "description": description_text,
                    "tags": tags_text,
                    "parent_channel": parent_text,
                    "channel": channel_text,
                    "title_terms": join_tokens(title_text),
                    "description_terms": join_tokens(description_text),
                    "tags_terms": join_tokens(tags_text),
                    "parent_terms": join_tokens(parent_text),
                    "channel_terms": join_tokens(channel_text),
                }
            )
            emitted += 1

        logging.info("prepared batch=%s rows=%s total_rows=%s", batch_index + 1, len(out_rows), emitted)
        yield pa.Table.from_pylist(out_rows)


def run_build(args: argparse.Namespace) -> None:
    dataset = ds.dataset(args.input.as_posix(), format="parquet")
    field_names = set(dataset.schema.names)
    title_column = resolve_column(field_names, args.title_column, DEFAULT_TITLE_CANDIDATES, "title")
    description_column = resolve_column(
        field_names,
        args.description_column,
        DEFAULT_DESCRIPTION_CANDIDATES,
        "description",
    )
    tags_column = resolve_column(field_names, args.tags_column, DEFAULT_TAGS_CANDIDATES, "tags")
    parent_column = resolve_column(field_names, args.parent_column, DEFAULT_PARENT_CANDIDATES, "parent partition")
    channel_column = resolve_column(field_names, args.channel_column, DEFAULT_CHANNEL_CANDIDATES, "channel partition")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db.as_posix())
    con.execute("LOAD fts")
    for sql in create_schema_sql(args.table, args.overwrite):
        con.execute(sql)

    inserted = 0
    for batch_index, table in enumerate(
        build_batches(
            input_path=args.input,
            batch_size=args.batch_size,
            limit_rows=args.limit_rows,
            title_column=title_column,
            description_column=description_column,
            tags_column=tags_column,
            parent_column=parent_column,
            channel_column=channel_column,
        )
    ):
        con.register("batch_docs", table)
        con.execute(f"INSERT INTO {args.table} SELECT * FROM batch_docs")
        con.unregister("batch_docs")
        inserted += table.num_rows
        if (batch_index + 1) % 10 == 0:
            logging.info("inserted batches=%s rows=%s", batch_index + 1, inserted)

    con.execute(
        f"""
        PRAGMA create_fts_index(
            '{args.table}',
            'id',
            'title_terms',
            'description_terms',
            'tags_terms',
            'parent_terms',
            'channel_terms',
            overwrite=1,
            stemmer='none',
            stopwords='none',
            ignore='(\\s+)',
            lower=0,
            strip_accents=0
        )
        """
    )
    row_count = con.execute(f"SELECT count(*) FROM {args.table}").fetchone()[0]
    print(
        json.dumps(
            {
                "db": args.db.as_posix(),
                "table": args.table,
                "rows": row_count,
                "title_column": title_column,
                "description_column": description_column,
                "tags_column": tags_column,
                "parent_column": parent_column,
                "channel_column": channel_column,
            },
            ensure_ascii=False,
        )
    )


def normalize_query_terms(terms: list[str], tokenize_query: bool) -> list[str]:
    if tokenize_query:
        tokens: list[str] = []
        for term in terms:
            tokens.extend(tokenize_cjk_text(term))
    else:
        tokens = [" ".join(term.strip().split()).lower() for term in terms if term.strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    if not deduped:
        raise ValueError("No non-empty query terms")
    return deduped


def run_query(args: argparse.Namespace) -> None:
    terms = normalize_query_terms(args.terms, args.tokenize_query)
    query_text = " ".join(terms)

    con = duckdb.connect(args.db.as_posix(), read_only=True)
    con.execute("LOAD fts")
    fts_schema = f"fts_main_{args.table}"
    sql = f"""
        WITH scored AS (
            SELECT
                id,
                title,
                description,
                tags,
                parent_channel,
                channel,
                {fts_schema}.match_bm25(id, ?, fields := 'title_terms', conjunctive := ?) AS title_score,
                {fts_schema}.match_bm25(id, ?, fields := 'description_terms', conjunctive := ?) AS description_score,
                {fts_schema}.match_bm25(id, ?, fields := 'tags_terms', conjunctive := ?) AS tags_score,
                {fts_schema}.match_bm25(id, ?, fields := 'parent_terms', conjunctive := ?) AS parent_score,
                {fts_schema}.match_bm25(id, ?, fields := 'channel_terms', conjunctive := ?) AS channel_score
            FROM {args.table}
        )
        SELECT
            id,
            title,
            description,
            tags,
            parent_channel,
            channel,
            title_score,
            description_score,
            tags_score,
            parent_score,
            channel_score,
            coalesce(title_score, 0) * ?
            + coalesce(description_score, 0) * ?
            + coalesce(tags_score, 0) * ?
            + coalesce(parent_score, 0) * ?
            + coalesce(channel_score, 0) * ? AS total_score
        FROM scored
        WHERE title_score IS NOT NULL
           OR description_score IS NOT NULL
           OR tags_score IS NOT NULL
           OR parent_score IS NOT NULL
           OR channel_score IS NOT NULL
        ORDER BY total_score DESC, id DESC
        LIMIT ?
    """
    bool_flag = 1 if args.conjunctive else 0
    params: list[object] = [
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        query_text,
        bool_flag,
        args.title_weight,
        args.description_weight,
        args.tags_weight,
        args.parent_weight,
        args.channel_weight,
        args.top_k,
    ]
    rows = con.execute(sql, params).fetchall()

    if args.output_parquet is not None:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            record: dict[str, object] = {
                "rank": index,
                "id": row[0],
                "query_text": query_text,
                "query_terms": terms,
                "total_score": float(row[11]),
                "title_score": None if row[6] is None else float(row[6]),
                "description_score": None if row[7] is None else float(row[7]),
                "tags_score": None if row[8] is None else float(row[8]),
                "parent_score": None if row[9] is None else float(row[9]),
                "channel_score": None if row[10] is None else float(row[10]),
            }
            if args.include_info_columns:
                record.update(
                    {
                        "title": row[1],
                        "description": row[2],
                        "tags": row[3],
                        "parent_channel": row[4],
                        "channel": row[5],
                    }
                )
            records.append(record)
        pd.DataFrame.from_records(records).to_parquet(args.output_parquet, index=False)
        logging.info("wrote query parquet to %s rows=%s", args.output_parquet, len(records))

    if args.json:
        payload = []
        for row in rows:
            payload.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "tags": row[3],
                    "parent_channel": row[4],
                    "channel": row[5],
                    "title_score": row[6],
                    "description_score": row[7],
                    "tags_score": row[8],
                    "parent_score": row[9],
                    "channel_score": row[10],
                    "total_score": row[11],
                }
            )
        print(
            json.dumps(
                {
                    "terms": terms,
                    "query_text": query_text,
                    "top_k": args.top_k,
                    "conjunctive": args.conjunctive,
                    "results": payload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"query_terms: {' '.join(terms)}")
    print(f"top_k: {args.top_k}")
    if not rows:
        print("results: 0")
        return
    for index, row in enumerate(rows, start=1):
        description = row[2] or ""
        if len(description) > args.show_description_chars:
            description = description[: args.show_description_chars] + "..."
        print(
            json.dumps(
                {
                    "rank": index,
                    "id": row[0],
                    "score": round(float(row[11]), 6),
                    "title": row[1],
                    "tags": row[3],
                    "parent_channel": row[4],
                    "channel": row[5],
                    "description": description,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    jieba.setLogLevel(logging.WARNING)
    if args.command == "build":
        run_build(args)
    else:
        run_query(args)


if __name__ == "__main__":
    main()
