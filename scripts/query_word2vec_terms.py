#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec


DEFAULT_MODEL = Path("data/acfun.videoinfo.20260307.full.word2vec.kv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query similar terms from a trained AcFun Word2Vec model."
    )
    parser.add_argument("terms", nargs="+", help="Seed terms")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to .kv or .model")
    parser.add_argument("--topn", type=int, default=10, help="Neighbors per seed term")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum cosine similarity")
    parser.add_argument("--keywords-only", action="store_true", help="Only print deduplicated keywords")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser.parse_args()


def load_vectors(path: Path) -> KeyedVectors:
    if path.suffix == ".model":
        return Word2Vec.load(path.as_posix()).wv
    return KeyedVectors.load(path.as_posix(), mmap="r")


def main() -> None:
    args = parse_args()
    vectors = load_vectors(args.model)

    results: dict[str, list[tuple[str, float]]] = {}
    keywords: list[str] = []
    seen: set[str] = set()

    for term in args.terms:
        if term in vectors:
            sims = [
                (token, float(score))
                for token, score in vectors.most_similar(term, topn=args.topn)
                if float(score) >= args.min_score
            ]
        else:
            sims = []
        results[term] = sims

        for token in [term, *[token for token, _ in sims]]:
            if token not in seen:
                seen.add(token)
                keywords.append(token)

    if args.keywords_only:
        print(" ".join(keywords))
        return

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model.as_posix(),
                    "keywords": keywords,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"model: {args.model}")
    print(f"keywords: {' '.join(keywords)}")
    for term, sims in results.items():
        if not sims:
            print(json.dumps({"term": term, "oov": True}, ensure_ascii=False))
            continue
        print(
            json.dumps(
                {
                    "term": term,
                    "neighbors": [{"term": token, "score": round(score, 6)} for token, score in sims],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
