"""The measurement itself.

Every parameter in config.py was chosen by this harness rather than by argument,
so a quiet fault here would not produce a visible failure - it would produce a
wrong configuration that looks measured.
"""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from eval import run_eval


def retrieved(*sources: str) -> list[Document]:
    return [Document(page_content="", metadata={"source": s}) for s in sources]


# ---------- the metric ----------

def test_the_rank_of_the_first_expected_page_is_one_based() -> None:
    docs = retrieved("other.md", "wanted.md", "another.md")
    assert run_eval.rank_of_first_hit(docs, ["wanted.md"]) == 2


def test_a_question_with_several_acceptable_pages_scores_the_earliest() -> None:
    """Some questions are answered by either of two pages, and ranking one of
    them first is a hit, not a partial one."""
    docs = retrieved("noise.md", "second-choice.md", "first-choice.md")
    assert run_eval.rank_of_first_hit(docs, ["first-choice.md", "second-choice.md"]) == 2


def test_a_question_with_no_expected_page_retrieved_is_a_miss() -> None:
    assert run_eval.rank_of_first_hit(retrieved("a.md", "b.md"), ["c.md"]) is None


def test_hit_rate_and_mrr_are_computed_over_every_question(monkeypatch) -> None:
    """MRR is the number that separates ranking the right page first from fifth,
    which hit@k cannot see. A miss contributes zero to both."""
    answers = {
        "first": retrieved("wanted.md"),
        "second": retrieved("noise.md", "wanted.md"),
        "missed": retrieved("noise.md"),
    }
    monkeypatch.setattr(run_eval, "retrieve", lambda q, **kwargs: answers[q])

    questions = [{"q": q, "expect": ["wanted.md"]} for q in answers]
    result = run_eval.score(questions, k=5, mode="hybrid", mmr=False, lambda_mult=0.0)

    assert result["hit_rate"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx((1 + 0.5) / 3)
    assert result["ranks"] == [1, 2, None]


def test_the_configuration_under_test_reaches_the_retriever(monkeypatch) -> None:
    """The sweep compares configurations, so one that is not passed through would
    make every row of the comparison identical and the conclusion meaningless."""
    seen = {}

    def capture(query, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(run_eval, "retrieve", capture)
    run_eval.score([{"q": "x", "expect": ["a.md"]}], 5, "dense", True, 0.7, rrf_k=42)

    assert seen == {"k": 5, "mode": "dense", "mmr": True, "lambda_mult": 0.7, "rrf_k": 42}


# ---------- the regression gate ----------

def test_a_result_that_clears_both_floors_reports_nothing() -> None:
    assert run_eval.below_floor({"hit_rate": 0.71, "mrr": 0.540}, 0.68, 0.50) == []


def test_a_result_at_the_floor_exactly_is_not_a_regression() -> None:
    """The floors sit below the measured numbers, so the question they answer is
    whether retrieval got worse, not whether it moved."""
    assert run_eval.below_floor({"hit_rate": 0.68, "mrr": 0.50}, 0.68, 0.50) == []


def test_each_metric_is_reported_with_the_floor_it_missed() -> None:
    """A CI log that says only "failed" sends the reader back to run it locally."""
    failures = run_eval.below_floor({"hit_rate": 0.61, "mrr": 0.40}, 0.68, 0.50)

    assert len(failures) == 2
    assert "0.610" in failures[0] and "0.680" in failures[0]
    assert failures[1].startswith("MRR")


def test_mrr_can_fall_while_hit_rate_holds() -> None:
    """The case for measuring both: the right page is still retrieved, but lower
    down, where it competes with four other chunks for the model's attention."""
    failures = run_eval.below_floor({"hit_rate": 0.71, "mrr": 0.40}, 0.68, 0.50)
    assert [f.split()[0] for f in failures] == ["MRR"]


def test_an_unset_floor_is_not_checked() -> None:
    """A local run reports the numbers without asserting anything about them."""
    assert run_eval.below_floor({"hit_rate": 0.10, "mrr": 0.05}, None, None) == []


def test_the_gate_is_off_unless_a_floor_is_given() -> None:
    assert run_eval.parse_args([]).min_hit is None
    assert run_eval.parse_args(["--min-hit", "0.68"]).min_hit == 0.68
    assert run_eval.parse_args(["--sweep"]).sweep is True


# ---------- the question set ----------

def test_an_expected_page_absent_from_the_corpus_is_a_broken_test(
    monkeypatch, tmp_path: Path
) -> None:
    """A path that no longer exists would be scored as a permanent miss and drag
    every future measurement down without ever being noticed."""
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{"q": "anything", "expect": ["bybit/gone.mdx"]}]), encoding="utf-8"
    )
    monkeypatch.setattr(run_eval, "QUESTIONS", questions)
    monkeypatch.setattr(run_eval, "RAW_DIR", tmp_path / "raw")

    with pytest.raises(SystemExit, match="absent from data/raw"):
        run_eval.load_questions()


def test_the_shipped_question_set_is_well_formed() -> None:
    """Checked without the corpus, so it runs on every push rather than only in
    the nightly job that rebuilds the index."""
    questions = json.loads(run_eval.QUESTIONS.read_text(encoding="utf-8"))

    assert len(questions) >= 60
    for item in questions:
        assert item["q"].strip()
        assert item["expect"], f"no expected page for {item['q']!r}"
        for source in item["expect"]:
            assert not source.startswith("/") and "\\" not in source


def test_no_question_is_asked_twice() -> None:
    """Duplicates weight one question twice in an average presented as an average
    over distinct cases."""
    questions = json.loads(run_eval.QUESTIONS.read_text(encoding="utf-8"))
    asked = [item["q"].strip().lower() for item in questions]

    assert len(set(asked)) == len(asked)
