"""Query preparation and rank fusion.

Nothing here touches the index. The two decisions taken before retrieval runs -
which exchange the question is about, and what the query should look like once
that name has done its job - were both worth more on the eval set than any
parameter tuned afterwards, so they are worth pinning down.
"""

import pytest
from langchain_core.documents import Document

import search


def docs(*texts: str) -> list[Document]:
    return [Document(page_content=text, metadata={}) for text in texts]


# ---------- which exchange the question is about ----------

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how do I change leverage on Bybit", "bybit"),
        ("bybit websocket authentication", "bybit"),
        ("how do I cancel an order on Binance", "binance-spot"),
        ("BINANCE rate limits", "binance-spot"),
    ],
)
def test_a_named_exchange_selects_that_corpus(query: str, expected: str) -> None:
    assert search.detect_exchange(query) == expected


def test_a_comparative_question_keeps_the_whole_corpus() -> None:
    """Filtering to one exchange would make the comparison impossible to answer."""
    assert search.detect_exchange("compare Binance and Bybit rate limits") is None


def test_a_generic_question_keeps_the_whole_corpus() -> None:
    assert search.detect_exchange("how do I cancel an order") is None


# ---------- what is left of the query afterwards ----------

def test_the_preposition_leaves_with_the_name() -> None:
    """Deleting the name alone leaves "how do I change leverage on", and the
    embedding of a truncated sentence is worse than either whole one: the correct
    page for this question falls from rank 1 to rank 9."""
    assert (
        search.strip_exchange_name("how do I change leverage on Bybit")
        == "how do I change leverage"
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how do I cancel an order on Binance", "how do I cancel an order"),
        ("Bybit rate limits for orders", "rate limits for orders"),
        ("what is the Binance API", "what is the API"),
        ("place an order on Bybit using python", "place an order using python"),
    ],
)
def test_the_query_stays_a_well_formed_sentence(query: str, expected: str) -> None:
    assert search.strip_exchange_name(query) == expected


def test_stripping_leaves_no_double_space_behind() -> None:
    assert "  " not in search.strip_exchange_name("cancel an order on Bybit quickly")


# ---------- fusion ----------

def test_agreement_between_the_two_arms_outranks_one_strong_vote() -> None:
    """The point of fusing at all: a page both dense and keyword search rank
    highly is a better answer than one only a single arm likes."""
    both = docs("agreed")[0]
    dense_only = docs("dense favourite")[0]

    fused = search.fuse([[dense_only, both], [docs("other")[0], both]], k=3, rrf_k=10)
    assert fused[0] is both


def test_rrf_k_decides_how_much_rank_counts() -> None:
    """Why 60 was not inherited. At 60 the curve is flat enough that a page both
    arms merely tolerate beats one a single arm is certain about; at 10 certainty
    still wins. Vocabulary-mismatch questions are exactly the case where only one
    arm is certain, which is what the keyword arm was added for."""
    tolerated = Document(page_content="ranked 20th by both arms", metadata={})
    certain = Document(page_content="ranked 1st by one arm", metadata={})

    dense = [certain, *docs(*[f"dense filler {i}" for i in range(18)]), tolerated]
    keyword = [*docs(*[f"keyword filler {i}" for i in range(19)]), tolerated]

    def order(rrf_k: int) -> tuple[int, int]:
        fused = search.fuse([dense, keyword], k=40, rrf_k=rrf_k)
        return fused.index(certain), fused.index(tolerated)

    sharp, flat = order(rrf_k=10), order(rrf_k=60)
    assert sharp[0] < sharp[1]
    assert flat[1] < flat[0]


def test_a_document_returned_by_both_arms_appears_once() -> None:
    """Both arms read the same chunks, so overlap is the normal case, not the
    exception. A duplicate would occupy two of the five context slots."""
    shared = Document(page_content="same chunk", metadata={})
    fused = search.fuse([[shared], [shared]], k=5)
    assert len(fused) == 1


def test_fusion_returns_at_most_k() -> None:
    assert len(search.fuse([docs("a", "b", "c", "d")], k=2)) == 2


def test_an_unknown_mode_is_refused(monkeypatch) -> None:
    """The mode reaches retrieve from a CLI flag and from the eval sweep."""
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        search.retrieve("anything", mode="semantic")


# ---------- the two decisions reaching the retrievers ----------

def test_both_arms_receive_the_filter_and_the_cleaned_query(monkeypatch) -> None:
    """Asserted at the boundary rather than through the index: the dense and
    keyword arms have to be given the same query, or fusing their ranks compares
    answers to two different questions."""
    seen: dict[str, tuple] = {}

    def fake_dense(query, k, exchange, mmr=False, lambda_mult=0.0):
        seen["dense"] = (query, exchange)
        return docs("dense result")

    def fake_keyword(query, k, exchange=None):
        seen["keyword"] = (query, exchange)
        return docs("keyword result")

    monkeypatch.setattr(search, "dense", fake_dense)
    monkeypatch.setattr(search.keyword_search, "search", fake_keyword)

    search.retrieve("how do I change leverage on Bybit", mode="hybrid")

    assert seen["dense"] == ("how do I change leverage", "bybit")
    assert seen["keyword"] == seen["dense"]
