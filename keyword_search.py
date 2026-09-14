"""Keyword retrieval over the same chunks, using BM25.

Dense retrieval matches meaning and is what makes "how do I cancel an order" find
a page titled "Cancel Order". It is weak in the opposite case: when the question
carries an exact term the document also uses, the signal gets averaged away with
everything else in the vector. Measured misses on this corpus were all of that
shape - "order does not exist" is written verbatim next to error code -2013, and
dense search still did not return that page.

BM25 is the classic answer: score by term overlap, weighted so rare terms count
for more and long documents are not rewarded for length alone.
"""

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from store import all_documents

# API reference text is full of tokens that whitespace splitting mangles:
# /v5/order/create-order, timeInForce, -2013, GTC. Splitting on non-alphanumerics
# and again on camelCase boundaries turns those into terms a question can match -
# a user typing "time in force" should reach a page that writes "timeInForce".
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NON_WORD = re.compile(r"[^A-Za-z0-9]+")

# Dropped from the query side only. IDF already discounts words that are common
# across the corpus, but not evenly: guides and FAQ-style pages are written in full
# sentences and so carry far more "how", "do" and "I" than a reference page of
# parameter tables. Scoring on those words ranked an IP changelog above the leverage
# reference for "how do I change leverage". Function words only - anything that
# could name a concept in an API stays in.
STOPWORDS = frozenset(
    """a an the i my me it its this that there and or if
    do does did how what when where which who why
    can could should would will is are was were be been am
    to of in on at for from with by about any some
    get use using way need want""".split()
)


def tokenize(text: str) -> list[str]:
    split = CAMEL_BOUNDARY.sub(" ", text)
    return [token for token in NON_WORD.split(split.lower()) if token]


def query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    content = [token for token in tokens if token not in STOPWORDS]
    # A question made entirely of function words would otherwise score nothing at
    # all; falling back to the raw tokens is worse than nothing only in theory.
    return content or tokens


class _Index:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in documents])

    def top(self, query: str, k: int) -> list[Document]:
        scores = self.bm25.get_scores(query_tokens(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # BM25 scores zero when no query term appears; returning those would pad
        # the result with arbitrary documents and pollute the fusion downstream.
        return [self.documents[i] for i in ranked[:k] if scores[i] > 0]


# One index per exchange filter. BM25 has no metadata filtering of its own, and
# building over the already-narrowed subset is both correct and cheaper than
# retrieving widely and discarding afterwards.
_indexes: dict[str | None, _Index] = {}


def _index_for(exchange: str | None) -> _Index:
    if exchange not in _indexes:
        documents = all_documents()
        if exchange:
            documents = [
                doc for doc in documents if doc.metadata["exchange"] == exchange
            ]
        _indexes[exchange] = _Index(documents)
    return _indexes[exchange]


def search(query: str, k: int, exchange: str | None = None) -> list[Document]:
    return _index_for(exchange).top(query, k)
