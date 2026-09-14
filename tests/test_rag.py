"""Grounded generation: what the model is shown, and what comes back.

No provider is called. The model is replaced by one that returns the prompt it
was given, which is the only way to assert that the passages and the grounding
rules actually reached it - the part that makes an answer checkable.
"""

from langchain_core.documents import Document
from langchain_core.language_models import SimpleChatModel

import rag


def passage(text: str, source: str = "bybit/docs/v5/order/create-order.mdx") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "exchange": source.split("/", 1)[0],
            "section": "Create Order > Request Parameters",
        },
    )


class EchoModel(SimpleChatModel):
    """Answers with the system prompt it received."""

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        return messages[0].content


def test_context_is_numbered_from_one_and_names_its_source() -> None:
    """The citation markers in the answer are only traceable if the numbering in
    the prompt matches the numbering the UI renders."""
    context = rag.format_context([passage("first"), passage("second")])

    assert context.startswith("[1]")
    assert "[2]" in context
    assert "exchange: bybit" in context
    assert "file: bybit/docs/v5/order/create-order.mdx" in context


def test_sources_are_described_in_the_order_they_were_retrieved() -> None:
    described = rag.describe([passage("first"), passage("second")])

    assert [item["n"] for item in described] == [1, 2]
    assert described[0]["excerpt"] == "first"
    assert described[0]["section"] == "Create Order > Request Parameters"


def test_an_excerpt_is_collapsed_and_bounded() -> None:
    """The excerpt is rendered in the UI so a reader can check the answer against
    what the model actually saw; raw chunk whitespace makes that unreadable."""
    described = rag.describe([passage("wrapped\n  text\n\n" + "long " * 400)])

    assert described[0]["excerpt"].startswith("wrapped text long")
    assert len(described[0]["excerpt"]) <= 600


def test_the_model_is_given_the_passages_and_the_grounding_rules(monkeypatch) -> None:
    monkeypatch.setattr(rag, "retrieve", lambda question: [passage("Set the leverage.")])
    monkeypatch.setattr(rag, "get_model", lambda: (EchoModel(), "test/echo"))

    result = rag.answer("how do I change leverage on Bybit")

    assert "Set the leverage." in result["answer"]
    assert "Do not fall back on prior knowledge" in result["answer"]
    assert "Never combine endpoints" in result["answer"]


def test_an_answer_reports_which_provider_produced_it(monkeypatch) -> None:
    """Two providers answer the same questions, and the eval numbers only mean
    something next to the label of what actually replied."""
    monkeypatch.setattr(rag, "retrieve", lambda question: [passage("body")])
    monkeypatch.setattr(rag, "get_model", lambda: (EchoModel(), "gemini/test-model"))

    result = rag.answer("anything")

    assert result["provider"] == "gemini/test-model"
    assert result["sources"][0]["source"].endswith("create-order.mdx")
