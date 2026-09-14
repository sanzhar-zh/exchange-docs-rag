"""Retrieval, with no model involved.

Generation can only cite what this returns, so it is worth being able to run and
inspect the retrieval step on its own.

    python search.py "how do I cancel an order"
    python search.py --mode dense "how do I cancel an order"
    python search.py --mode keyword "order does not exist"
"""

import re
import sys
import textwrap

from langchain_core.documents import Document

import keyword_search
from config import (
    FETCH_K,
    MMR_LAMBDA,
    RETRIEVAL_MODE,
    RRF_K,
    TOP_K,
    USE_MMR,
)
from store import get_store

MODES = ("hybrid", "dense", "keyword")

# Dense retrieval ranks by meaning, and "cancel an order on Binance" reads as very
# similar to the same sentence about Bybit, so a question naming one exchange still
# pulls chunks from the other and wastes context. Where the question is explicit,
# narrowing the search space first is cheaper and more reliable than hoping the
# embedding keeps them apart.
EXCHANGE_ALIASES = {
    "binance-spot": ("binance",),
    "bybit": ("bybit",),
}

# The name is taken out together with the preposition that introduces it, because
# deleting it alone leaves a dangling fragment - "how do I change leverage on" -
# and the embedding of a truncated sentence is worse than the embedding of either
# the full question or a cleanly trimmed one. On the eval set that difference is
# not small: the correct page for that example falls from rank 1 to rank 9 when the
# preposition is left behind, and removing it too is worth 0.022 hit@5 overall.
EXCHANGE_NAME = re.compile(
    r"(?i)\s*\b(?:on|for|in|at|with|from|of|to)\s+(?:bybit|binance)\b"
    r"|\s*\b(?:bybit|binance)\b"
)
EXTRA_SPACE = re.compile(r"\s{2,}")


def strip_exchange_name(query: str) -> str:
    """The question with the exchange name, and its preposition, taken out."""
    return EXTRA_SPACE.sub(" ", EXCHANGE_NAME.sub(" ", query)).strip()


def detect_exchange(query: str) -> str | None:
    """The exchange named in the question, if exactly one is.

    Two exchanges, or none, means the question is comparative or generic and the
    whole corpus should stay in play.
    """
    lowered = query.lower()
    named = [
        exchange
        for exchange, aliases in EXCHANGE_ALIASES.items()
        if any(alias in lowered for alias in aliases)
    ]
    return named[0] if len(named) == 1 else None


def dense(
    query: str,
    k: int,
    exchange: str | None,
    mmr: bool = USE_MMR,
    lambda_mult: float = MMR_LAMBDA,
) -> list[Document]:
    store = get_store()
    where = {"exchange": exchange} if exchange else None

    if mmr:
        return store.max_marginal_relevance_search(
            query, k=k, fetch_k=max(k, FETCH_K), lambda_mult=lambda_mult, filter=where
        )
    return [
        doc
        for doc, _ in store.similarity_search_with_relevance_scores(
            query, k=k, filter=where
        )
    ]


def fuse(
    rankings: list[list[Document]], k: int, rrf_k: int = RRF_K
) -> list[Document]:
    """Reciprocal Rank Fusion of several ranked lists.

    Each list contributes 1/(rrf_k + rank) to a document's score. Rank is used
    rather than the underlying score because cosine similarity and BM25 are not
    on a comparable scale and cannot be normalised into one without inventing a
    weighting; positions can be compared directly.

    rrf_k controls how sharply rank matters, and it has to suit the length of the
    lists being fused. The conventional 60 comes from TREC runs a thousand
    documents deep; over lists of 25 it flattens the curve so far that first place
    beats last by under a third, and a document both arms merely tolerate outranks
    one that a single arm is certain about. Vocabulary-mismatch questions are
    exactly the case where only one arm is certain, so the value is measured here
    rather than inherited - see eval/.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
            documents[key] = doc

    best = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [documents[key] for key in best[:k]]


def retrieve(
    query: str,
    k: int = TOP_K,
    mode: str = RETRIEVAL_MODE,
    mmr: bool = USE_MMR,
    lambda_mult: float = MMR_LAMBDA,
    rrf_k: int = RRF_K,
) -> list[Document]:
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}, expected one of {MODES}")

    exchange = detect_exchange(query)

    # Once the name has selected the corpus it should not also be searched for.
    # Within one exchange's documentation the brand carries no information, but it
    # does appear throughout that exchange's overview and marketing pages, which
    # then outrank the endpoint the question is actually about. Removing it after
    # filtering was worth 0.068 hit@5 on the eval set.
    if exchange:
        query = strip_exchange_name(query)

    if mode == "dense":
        return dense(query, k, exchange, mmr, lambda_mult)
    if mode == "keyword":
        return keyword_search.search(query, k, exchange)

    # Each arm contributes a deeper list than the final k: fusion needs candidates
    # below the cut to be able to promote something both arms agree on.
    return fuse(
        [
            dense(query, FETCH_K, exchange, mmr, lambda_mult),
            keyword_search.search(query, FETCH_K, exchange),
        ],
        k,
        rrf_k,
    )


def main() -> None:
    args = sys.argv[1:]
    mode = RETRIEVAL_MODE
    if len(args) >= 2 and args[0] == "--mode":
        mode, args = args[1], args[2:]
    if not args:
        raise SystemExit(
            'usage: python search.py [--mode hybrid|dense|keyword] "question"'
        )

    query = " ".join(args)
    exchange = detect_exchange(query)
    print(f"query: {query}\nmode: {mode}   exchange filter: {exchange or 'none'}\n")

    for i, doc in enumerate(retrieve(query, mode=mode), 1):
        meta = doc.metadata
        print(f"[{i}] {meta['exchange']}  {meta['section']}")
        print(f"    {meta['source']}")
        body = " ".join(doc.page_content.split())
        print(
            textwrap.fill(
                body[:280], width=96, initial_indent="    ", subsequent_indent="    "
            )
        )
        print()


if __name__ == "__main__":
    main()
