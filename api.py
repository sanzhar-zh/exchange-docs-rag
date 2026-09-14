"""HTTP API behind the web interface.

    uvicorn api:app --reload --port 8000

The embedding model and the BM25 index are built on first use and then held for
the process lifetime, so the first request after start is slow and the rest are
not. Warming them at startup moves that cost off the first user.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import answer
from search import retrieve

log = logging.getLogger("uvicorn.error")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # One throwaway retrieval loads the embedding model and builds the keyword
    # index, which together take a few seconds.
    retrieve("warm up the retrievers")
    yield


app = FastAPI(title="Exchange Docs RAG", lifespan=lifespan)

# The dev frontend runs on another port, so the browser treats it as cross-origin.
# Local development only; a deployment would serve both from one origin instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question) -> dict:
    try:
        return answer(payload.question)
    except Exception as failure:
        # Provider failures - an exhausted quota, a rejected key, a network
        # timeout - are the normal way this endpoint breaks, and letting them
        # escape is worse than it looks: Starlette's 500 handler sits outside the
        # CORS middleware, so the browser rejects the response as a CORS error and
        # the page reports the server as unreachable instead of quota-exhausted.
        # Answering with a handled status keeps the headers and the real reason.
        log.exception("generation failed")
        detail = str(failure)
        if "RESOURCE_EXHAUSTED" in detail or "429" in detail:
            raise HTTPException(
                status_code=429,
                detail="the model provider's quota is exhausted - switch "
                "LLM_PROVIDER in .env, or wait for the quota to reset",
            ) from failure
        raise HTTPException(
            status_code=502, detail=f"the model provider failed: {detail[:300]}"
        ) from failure
