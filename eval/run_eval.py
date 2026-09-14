"""Measure retrieval quality against a fixed question set.

The generation step can only cite what retrieval hands it, so this measures
retrieval alone: no model is called and a run costs nothing. That is what makes
it usable as a sweep - every configuration can be tried, not argued about.

    python eval/run_eval.py              current settings, per-question detail
    python eval/run_eval.py --sweep      compare search modes and MMR lambda

Two numbers, because they answer different questions:

  hit@k  the share of questions where an expected page appears anywhere in the
         top k. This is what decides whether the model can possibly be right.

  MRR    mean reciprocal rank: 1/rank of the first expected page, averaged, and 0
         for a miss. Rewards ranking the right page first rather than fifth, which
         hit@k cannot see. A config can hold hit@k steady while MRR falls, and that
         still costs quality: lower-ranked context competes with four other chunks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, RRF_K, TOP_K
from search import retrieve

QUESTIONS = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    """Load the set, refusing to score against pages that are not in the corpus.

    An expected path that no longer exists would be counted as a permanent miss
    and quietly drag every measurement down, so it is treated as a broken test
    rather than a bad result.
    """
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    missing = {
        source
        for item in questions
        for source in item["expect"]
        if not (RAW_DIR / source).exists()
    }
    if missing:
        raise SystemExit(
            "expected sources absent from data/raw:\n  "
            + "\n  ".join(sorted(missing))
            + "\nre-run scripts/fetch_docs.py, or fix the paths in questions.json"
        )
    return questions


def rank_of_first_hit(docs: list, expected: list[str]) -> int | None:
    """1-based position of the first retrieved chunk from an expected page."""
    wanted = set(expected)
    for position, doc in enumerate(docs, 1):
        if doc.metadata["source"] in wanted:
            return position
    return None


def score(
    questions: list[dict],
    k: int,
    mode: str,
    mmr: bool,
    lambda_mult: float,
    rrf_k: int = RRF_K,
) -> dict:
    ranks: list[int | None] = []
    for item in questions:
        docs = retrieve(
            item["q"],
            k=k,
            mode=mode,
            mmr=mmr,
            lambda_mult=lambda_mult,
            rrf_k=rrf_k,
        )
        ranks.append(rank_of_first_hit(docs, item["expect"]))

    hits = [r for r in ranks if r is not None]
    return {
        "hit_rate": len(hits) / len(questions),
        "mrr": sum(1 / r for r in hits) / len(questions),
        "ranks": ranks,
    }


def report_detail(questions: list[dict], result: dict) -> None:
    for item, rank in zip(questions, result["ranks"]):
        mark = f"#{rank}" if rank else "MISS"
        print(f"  {mark:>5}  {item['q']}")
        if rank is None:
            print(f"         expected: {', '.join(item['expect'])}")


def main() -> None:
    questions = load_questions()
    sweep = "--sweep" in sys.argv

    if not sweep:
        from config import MMR_LAMBDA, RETRIEVAL_MODE, USE_MMR

        result = score(questions, TOP_K, RETRIEVAL_MODE, USE_MMR, MMR_LAMBDA)
        dense_mode = f"mmr l={MMR_LAMBDA}" if USE_MMR else "similarity"
        print(
            f"{len(questions)} questions, k={TOP_K}, "
            f"{RETRIEVAL_MODE} ({dense_mode})\n"
        )
        report_detail(questions, result)
        print(f"\nhit@{TOP_K} {result['hit_rate']:.2f}   MRR {result['mrr']:.3f}")
        return

    configs = [
        ("dense similarity", "dense", False, 0.0, RRF_K),
        ("dense mmr l=0.7", "dense", True, 0.7, RRF_K),
        ("keyword bm25", "keyword", False, 0.0, RRF_K),
    ] + [
        (f"hybrid rrf_k={r}", "hybrid", True, 0.7, r)
        for r in (1, 2, 5, 10, 20, 40, 60)
    ]

    print(f"{len(questions)} questions, k={TOP_K}\n")
    print(f"{'config':<19} {'hit@' + str(TOP_K):>7} {'MRR':>7}  {'misses':>6}")
    for name, mode, mmr, lambda_mult, rrf_k in configs:
        result = score(questions, TOP_K, mode, mmr, lambda_mult, rrf_k)
        misses = sum(1 for r in result["ranks"] if r is None)
        print(
            f"{name:<19} {result['hit_rate']:>7.2f} {result['mrr']:>7.3f}  {misses:>6}"
        )


if __name__ == "__main__":
    main()
