#!/usr/bin/env python3
import argparse
import logging
import os
import re
from pathlib import Path

import jieba
import pyarrow as pa
import pyarrow.dataset as ds
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer as JanomeTokenizer


DEFAULT_INPUT = Path("data/acfun.videoinfo.20260307.full.flattened.parquet")
DEFAULT_OUTPUT = Path("data/acfun.videoinfo.20260307.full.word2vec.model")
DEFAULT_TITLE_CANDIDATES = ("payload_title", "title")
DEFAULT_DESCRIPTION_CANDIDATES = ("payload_description", "description")

JANOME_TOKENIZER: JanomeTokenizer | None = None

LATIN_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[._+-][0-9a-z]+)*")
PUNCT_ONLY_RE = re.compile(r"^[^\w\u3400-\u9fff\u3040-\u30ff]+$")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
TOKEN_CHUNK_RE = re.compile(r"[0-9a-z]+(?:[._+-][0-9a-z]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gensim Word2Vec model from title and description fields in a parquet dataset."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help="Input parquet path",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Output model path",
    )
    parser.add_argument("--title-column", default=None, help="Title column name. Defaults to auto-detect.")
    parser.add_argument(
        "--description-column",
        default=None,
        help="Description column name. Defaults to auto-detect.",
    )
    parser.add_argument("--batch-size", type=int, default=20_000, help="Rows per parquet scan batch")
    parser.add_argument("--vector-size", type=int, default=256, help="Embedding size")
    parser.add_argument("--window", type=int, default=8, help="Context window size")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum token count")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Training workers")
    parser.add_argument("--sg", type=int, choices=(0, 1), default=1, help="1=skip-gram, 0=CBOW")
    parser.add_argument("--negative", type=int, default=10, help="Negative samples")
    parser.add_argument("--sample", type=float, default=1e-5, help="Downsampling rate for frequent tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hs", type=int, choices=(0, 1), default=0, help="Use hierarchical softmax")
    parser.add_argument("--max-final-vocab", type=int, default=None, help="Optional vocab cap")
    parser.add_argument(
        "--progress-per",
        type=int,
        default=100_000,
        help="Log progress every N sentences during vocabulary building",
    )
    parser.add_argument(
        "--report-delay",
        type=float,
        default=10.0,
        help="Training progress report interval in seconds",
    )
    parser.add_argument(
        "--save-vectors",
        type=Path,
        default=None,
        help="Optional KeyedVectors output path",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Only use the first N rows, useful for smoke tests",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level",
    )
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


class ParquetSentenceIterator:
    def __init__(
        self,
        dataset: ds.Dataset,
        title_column: str,
        description_column: str,
        batch_size: int,
        limit_rows: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.title_column = title_column
        self.description_column = description_column
        self.batch_size = batch_size
        self.limit_rows = limit_rows

    def __iter__(self):
        scanner = self.dataset.scanner(
            columns=[self.title_column, self.description_column],
            batch_size=self.batch_size,
            use_threads=True,
        )
        emitted = 0
        for batch_index, batch in enumerate(scanner.to_batches()):
            titles = batch.column(0).to_pylist()
            descriptions = batch.column(1).to_pylist()
            for title, description in zip(titles, descriptions, strict=True):
                if self.limit_rows is not None and emitted >= self.limit_rows:
                    return
                text = f"{title or ''}\n{description or ''}"
                tokens = tokenize_cjk_text(text)
                if tokens:
                    yield tokens
                emitted += 1
            if (batch_index + 1) % 20 == 0:
                logging.info("scanned %s batches, %s rows", batch_index + 1, emitted)


def row_count(dataset: ds.Dataset, limit_rows: int | None) -> int:
    total = dataset.count_rows()
    if limit_rows is not None:
        return min(total, limit_rows)
    return total


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset = ds.dataset(args.input.as_posix(), format="parquet")
    field_names = set(dataset.schema.names)
    title_column = resolve_column(field_names, args.title_column, DEFAULT_TITLE_CANDIDATES, "title")
    description_column = resolve_column(
        field_names,
        args.description_column,
        DEFAULT_DESCRIPTION_CANDIDATES,
        "description",
    )
    total_rows = row_count(dataset, args.limit_rows)
    sentences = ParquetSentenceIterator(
        dataset=dataset,
        title_column=title_column,
        description_column=description_column,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows,
    )

    logging.info(
        "training word2vec input=%s rows=%s title=%s description=%s output=%s",
        args.input,
        total_rows,
        title_column,
        description_column,
        args.output,
    )

    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        negative=args.negative,
        sample=args.sample,
        seed=args.seed,
        hs=args.hs,
        max_final_vocab=args.max_final_vocab,
    )
    model.build_vocab(sentences, progress_per=args.progress_per)
    logging.info("vocab size=%s corpus_count=%s", len(model.wv), model.corpus_count)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=args.epochs,
        report_delay=args.report_delay,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output.as_posix())
    logging.info("saved model to %s", args.output)

    if args.save_vectors is not None:
        args.save_vectors.parent.mkdir(parents=True, exist_ok=True)
        model.wv.save(args.save_vectors.as_posix())
        logging.info("saved keyed vectors to %s", args.save_vectors)


if __name__ == "__main__":
    main()
