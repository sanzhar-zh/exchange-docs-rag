"""Tokenisation and BM25 ranking.

The keyword arm exists to catch questions that share exact vocabulary with the
answer, so the tokeniser is the whole mechanism: a term the question and the page
write differently never meets, however good the ranking function is.
"""

import pytest
from langchain_core.documents import Document

import keyword_search


@pytest.fixture(autouse=True)
def clear_index_cache():
    """The module memoises one BM25 index per exchange filter for the process."""
    keyword_search._indexes.clear()
    yield
    keyword_search._indexes.clear()


def test_tokenize_splits_camel_case() -> None:
    """A user types "time in force"; the documentation writes "timeInForce"."""
    assert keyword_search.tokenize("timeInForce") == ["time", "in", "force"]
    assert keyword_search.tokenize("orderLinkId") == ["order", "link", "id"]


def test_tokenize_splits_endpoint_paths() -> None:
    assert keyword_search.tokenize("/v5/order/create-order") == [
        "v5",
        "order",
        "create",
        "order",
    ]


def test_tokenize_keeps_error_codes_as_terms() -> None:
    """Error codes are the most discriminating term a question can carry."""
    assert keyword_search.tokenize("error -2013 means") == ["error", "2013", "means"]


def test_tokenize_lowercases_acronyms() -> None:
    assert keyword_search.tokenize("GTC and IOC") == ["gtc", "and", "ioc"]


def test_a_spelled_out_question_meets_a_camel_case_parameter() -> None:
    """The case the tokeniser is for, asserted end to end over both sides."""
    document = set(keyword_search.tokenize("The timeInForce field accepts GTC."))
    query = set(keyword_search.query_tokens("what does time in force mean"))
    assert query & document == {"time", "force"}


def test_query_tokens_drop_function_words() -> None:
    """Guides are written in sentences and reference pages are not, so scoring on
    "how do I" ranked a changelog above the leverage reference."""
    assert keyword_search.query_tokens("how do I change leverage") == [
        "change",
        "leverage",
    ]


def test_query_tokens_keep_words_that_name_api_concepts() -> None:
    assert "position" in keyword_search.query_tokens("what is a position")
    assert "order" in keyword_search.query_tokens("how do I cancel an order")


def test_query_of_only_function_words_falls_back_to_the_raw_tokens() -> None:
    """Scoring nothing at all would return an arbitrary page rather than none."""
    assert keyword_search.query_tokens("how do I") == ["how", "do", "i"]


def corpus(exchange: str, *texts: str) -> list[Document]:
    """A handful of chunks from one exchange.

    Deliberately not two or three documents: BM25 weights a term by how rare it
    is, and in a corpus that small a term present in half the documents is scored
    as carrying no information at all. The real index holds five thousand chunks,
    so a fixture that small would be testing the corner rather than the code.
    """
    return [
        Document(page_content=text, metadata={"exchange": exchange}) for text in texts
    ]


PAGES = (
    "Set the leverage for a position.",
    "Cancel an existing order.",
    "Query the account balance.",
    "Place a new order on the book.",
    "Amend the price of an open order.",
    "Subscribe to public trade updates.",
)


def test_ranking_prefers_the_document_that_uses_the_term() -> None:
    index = keyword_search._Index(corpus("bybit", *PAGES))
    assert index.top("leverage", k=3)[0].page_content.startswith("Set the leverage")


def test_documents_matching_no_query_term_are_not_returned() -> None:
    """BM25 scores zero on no overlap; padding the list to k would put arbitrary
    pages into the fusion downstream, where a second weak vote can promote them."""
    index = keyword_search._Index(corpus("bybit", *PAGES))
    assert len(index.top("leverage", k=5)) == 1


def test_search_builds_its_index_over_one_exchange_only(monkeypatch) -> None:
    """BM25 cannot filter by metadata, so the filter has to narrow the corpus
    before the index is built rather than the results after."""
    both = corpus("bybit", *PAGES) + corpus(
        "binance-spot", *(text.replace(".", " here.") for text in PAGES)
    )
    monkeypatch.setattr(keyword_search, "all_documents", lambda: both)

    results = keyword_search.search("leverage", k=5, exchange="bybit")
    assert results
    assert {doc.metadata["exchange"] for doc in results} == {"bybit"}
