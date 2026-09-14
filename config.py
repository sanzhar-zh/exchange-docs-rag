"""Shared settings for the ingestion and query sides.

Both must agree on EMBEDDING_MODEL and CHROMA_DIR: a question is only comparable
to indexed chunks if it was embedded by the same model.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "exchange_docs"

# Runs locally on CPU, 384 dimensions, no API key and no per-token cost.
# Anthropic does not expose an embeddings endpoint, so the embedder has to come
# from somewhere else; keeping it local also means the corpus never leaves the machine.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Roughly 200-300 tokens per chunk. Small enough that a hit is specific,
# large enough to keep a whole endpoint description in one piece.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Generation is the only vendor-dependent part of this project: embeddings run
# locally, so retrieval works the same whichever provider answers. Preference
# order is this value first, then whichever provider actually has a key.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

# Opus 5 at $5/$25 per 1M tokens. Sonnet 5 ("claude-sonnet-5", $2/$10) is the
# cheaper option and adequate for grounded lookup, where the model summarises
# retrieved text rather than reasoning from scratch.
ANTHROPIC_MODEL = "claude-opus-5"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

# Answers about an endpoint routinely run to a dozen bullet points once the model
# has five passages of parameter tables in front of it. At 1024 they were being cut
# off mid-sentence, which reads as a bug in the assistant rather than a limit.
ANSWER_MAX_TOKENS = 2048

# Chunks handed to the model. Five fits comfortably in the prompt and keeps the
# answer traceable; more mostly adds near-duplicates.
TOP_K = 5

# Maximal Marginal Relevance: take FETCH_K candidates by similarity, then greedily
# pick TOP_K that are relevant to the query but unlike each other. Exact-hash
# deduplication at ingest time cannot catch pages that differ only by a hostname,
# and those otherwise occupy two slots with one piece of information.
# lambda_mult 1.0 is pure relevance, 0.0 is pure diversity.
FETCH_K = 25

# MMR re-ranks for diversity. Kept switchable because it is the obvious thing to
# reach for against near-duplicate results, but measurement says it earns nothing
# here: identical hit@5 and MRR to three decimals with it on and off, once fusion
# is in place. It costs a pass over FETCH_K candidates, so it is off.
USE_MMR = False
MMR_LAMBDA = 0.7

# "hybrid" fuses dense and keyword results, "dense" and "keyword" run one arm only.
# Dense alone missed every question whose wording differed from the documentation's
# while the exact phrase sat in the text; keyword alone cannot answer a question
# that shares no vocabulary with its answer.
RETRIEVAL_MODE = "hybrid"

# Reciprocal Rank Fusion constant. The conventional default is 60, from experiments
# on result lists a thousand deep; over lists of 25 that flattens the curve so far
# that rank barely matters. Measured across 1-60: hit@5 is flat from 10 upwards and
# MRR peaks at 10, so the sharper curve is kept.
RRF_K = 10

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
