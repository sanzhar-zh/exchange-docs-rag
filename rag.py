"""Retrieval plus grounded generation, shared by the CLI and the HTTP API.

The prompt is the whole safety mechanism here. Each rule below exists because of a
specific failure: a model that answers from memory cannot be checked, an answer
without citations cannot be traced, and a corpus holding two exchanges will happily
answer about one using the other's parameter names, since those passages look
nearly identical to an embedding.
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import get_model
from search import retrieve

SYSTEM = """You answer questions about cryptocurrency exchange REST and WebSocket APIs, \
using only the numbered sources below.

Rules:
- Use only what the sources state. Do not fall back on prior knowledge of these APIs.
- Cite the source number in square brackets after each claim, like [2].
- The sources cover two different exchanges. Never combine endpoints, parameter names \
or limits from one exchange into an answer about the other, and always say which \
exchange a fact belongs to.
- If the sources do not answer the question, say so plainly and name what is missing. \
Do not guess an endpoint or a parameter.

Sources:
{context}"""


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] exchange: {doc.metadata['exchange']} | "
        f"section: {doc.metadata['section']} | file: {doc.metadata['source']}\n"
        f"{doc.page_content}"
        for i, doc in enumerate(documents, 1)
    )


def describe(documents: list[Document]) -> list[dict]:
    """Retrieved passages in the shape the UI renders them.

    The excerpt is included so a reader can judge the answer against what the model
    was actually given, rather than trusting the citation number alone.
    """
    return [
        {
            "n": i,
            "exchange": doc.metadata["exchange"],
            "section": doc.metadata["section"],
            "source": doc.metadata["source"],
            "excerpt": " ".join(doc.page_content.split())[:600],
        }
        for i, doc in enumerate(documents, 1)
    ]


def answer(question: str) -> dict:
    documents = retrieve(question)
    model, provider = get_model()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("human", "{question}")]
    )
    chain = prompt | model | StrOutputParser()
    text = chain.invoke(
        {"context": format_context(documents), "question": question}
    )

    return {
        "answer": text,
        "sources": describe(documents),
        "provider": provider,
    }
