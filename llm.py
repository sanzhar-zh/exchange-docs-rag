"""Provider-agnostic access to the answering model.

Retrieval in this project is vendor-independent by construction: embeddings are
computed locally, so the index and the search behave identically whoever answers.
Generation is the only part that talks to a vendor, so it is the only part that
needs an abstraction.

Both providers are reached through LangChain's chat interface, so callers receive
the same object either way and never learn which one replied.

Selection order: LLM_PROVIDER if that provider has a key, otherwise whichever
provider does. Falling back rather than failing means a missing key degrades the
demo instead of stopping it.
"""

from langchain_core.language_models import BaseChatModel

from config import (
    ANSWER_MAX_TOKENS,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
)

KEYS = {"anthropic": ANTHROPIC_API_KEY, "gemini": GEMINI_API_KEY}


def provider_available(name: str) -> bool:
    return bool(KEYS.get(name))


def resolve_provider() -> str:
    if provider_available(LLM_PROVIDER):
        return LLM_PROVIDER

    for name in KEYS:
        if provider_available(name):
            return name

    raise SystemExit(
        "no provider key found - set ANTHROPIC_API_KEY or GEMINI_API_KEY in .env"
    )


def get_model() -> tuple[BaseChatModel, str]:
    """Returns the chat model and a label naming what actually answered."""
    provider = resolve_provider()

    # temperature 0 on both: this is lookup, not composition. The same question
    # should give the same answer, which is also what makes eval/ meaningful.
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=0,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=ANSWER_MAX_TOKENS,
        )
        return model, f"anthropic/{ANTHROPIC_MODEL}"

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=GEMINI_API_KEY,
        max_output_tokens=ANSWER_MAX_TOKENS,
    )
    return model, f"gemini/{GEMINI_MODEL}"
