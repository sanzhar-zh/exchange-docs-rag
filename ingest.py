"""Build the vector index from the documentation under data/raw/.

Two corpora with different shapes. Binance ships plain markdown. Bybit ships a
Docusaurus site: YAML frontmatter, JSX imports, admonition markers and inline
HTML, plus a Chinese translation and a machine-generated API explorer that are
both excluded below.

Two-stage split. Markdown headers first, so a chunk never spans two unrelated
endpoints and the section path survives into the metadata; then a size split,
because some sections run for several pages.

Re-running rebuilds the collection from scratch. The corpus is small enough that
incremental updates are not worth the bookkeeping.
"""

import hashlib
import re
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    EMBEDDING_MODEL,
    RAW_DIR,
)

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Binance ships every page twice, English and Chinese, and Bybit keeps its
# translation under i18n/. Indexing both languages pollutes retrieval: the
# translated copy of the right page competes with the English copy of a slightly
# less relevant one, for no gain on an English-language corpus.
SKIP_SUFFIXES = ("_CN.md",)

# faq.mdx is Bybit's single general FAQ: one very long page that touches every
# topic, which is exactly what makes it outrank the specific endpoint page for
# almost any question. Binance's faqs/ directory is different - those are separate
# focused documents and several are the correct answer to a question here.
SKIP_NAMES = {"CHANGELOG.md", "PROD-TERMS-OF-USE.md", "LICENSE.md", "faq.mdx"}

# api-explorer/ is generated from the OpenAPI spec: a few lines of prose wrapped
# in several kilobytes of JSX and JSON. The same endpoints are documented by hand
# under docs/v5/, which is what we want to retrieve from.
#
# testnet/ and demo-mode/ mirror the production reference against a different host
# and different limits. Deduplication alone was not enough: it keeps whichever copy
# sorts first, and "testnet/web-socket-api.md" sorts before "web-socket-api.md", so
# the sandbox rate limits were being indexed and the production ones dropped. This
# assistant answers about the production API, so the mirrors are excluded outright.
#
# changelog/ is a dated log of API changes. It names every endpoint in the product
# and so matches almost any question, while never being the answer to "how do I".
# Excluding it and faq.mdx was worth 0.027 MRR on the eval set; excluding the other
# broad pages (demo, guide, tradfi) on top of that measured as no change, so they
# stay in.
SKIP_DIRS = {
    "i18n",
    "api-explorer",
    "node_modules",
    "testnet",
    "demo-mode",
    "changelog",
}

FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
TITLE_RE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*import\s+.+?from\s+[\"'].+?[\"'];?\s*$", re.MULTILINE)
ADMONITION_RE = re.compile(r"^:::\w*\s*$", re.MULTILINE)
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,300}>")
BLANK_RUN_RE = re.compile(r"\n{3,}")


def collect_files() -> list:
    files = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.suffix not in (".md", ".mdx") or not path.is_file():
            continue
        parts = set(path.relative_to(RAW_DIR).parts)
        if parts & SKIP_DIRS or ".git" in parts:
            continue
        if path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES):
            continue
        files.append(path)
    return files


def clean(text: str) -> tuple[str, str]:
    """Strip Docusaurus scaffolding. Returns the body and the frontmatter title."""
    title = ""
    match = FRONTMATTER_RE.match(text)
    if match:
        found = TITLE_RE.search(match.group(1))
        title = found.group(1).strip() if found else ""
        text = text[match.end() :]

    text = IMPORT_RE.sub("", text)
    text = ADMONITION_RE.sub("", text)
    # Inline HTML carries no meaning here, but the prose between the tags does.
    # Code samples in this corpus are JSON and Python, so no generics get eaten.
    text = HTML_TAG_RE.sub("", text)
    return BLANK_RUN_RE.sub("\n\n", text).strip(), title


def split_file(path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text, title = clean(raw)
    if len(text) < 80:
        return []

    source = str(path.relative_to(RAW_DIR)).replace("\\", "/")
    exchange = source.split("/", 1)[0]

    # Pages whose topic lives only in the frontmatter get it back as an H1, so the
    # header splitter has something to anchor on and the topic reaches the vector.
    if title and not text.lstrip().startswith("#"):
        text = f"# {title}\n\n{text}"

    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks = size_splitter.split_documents(header_splitter.split_text(text))

    for chunk in chunks:
        section = " > ".join(
            chunk.metadata[k] for k in ("h1", "h2", "h3") if chunk.metadata.get(k)
        )
        chunk.metadata = {
            "source": source,
            "exchange": exchange,
            "section": section or title or path.stem,
        }
    return chunks


def deduplicate(documents: list[Document]) -> list[Document]:
    """Drop chunks whose text was already indexed.

    Binance mirrors its whole reference under testnet/, so roughly a fifth of the
    chunks exist twice. Duplicates do not just waste space: identical texts score
    identically, so they occupy adjacent slots in the top-k and push genuinely
    different pages out of the context window.

    Exact hashing only. Near-duplicates that differ by a hostname survive and are
    a known limitation.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)

    dropped = len(documents) - len(unique)
    print(f"deduplicated: -{dropped} -> {len(unique)}")
    return unique


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit("data/raw is empty - run scripts/fetch_docs.py first")

    files = collect_files()
    by_exchange: dict[str, int] = {}
    for path in files:
        key = path.relative_to(RAW_DIR).parts[0]
        by_exchange[key] = by_exchange.get(key, 0) + 1
    print(f"files: {len(files)} {by_exchange}")

    documents: list[Document] = []
    for path in files:
        documents.extend(split_file(path))
    print(f"chunks: {len(documents)}")

    documents = deduplicate(documents)

    if not documents:
        raise SystemExit("nothing to index")

    if CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError as locked:
            # Windows refuses to unlink a file another process has open, and the
            # API server holds the index for its whole lifetime. The raw traceback
            # points at shutil and not at the cause, so say what it is.
            raise SystemExit(
                f"cannot replace the index: {locked.filename}\n"
                "something still has it open - stop the API server (uvicorn) and "
                "any Python session holding it, then run this again"
            ) from locked
    CHROMA_DIR.mkdir(parents=True)

    print(f"embedding with {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        # Vectors are normalised, so cosine and L2 rank identically; cosine just
        # makes the reported scores directly interpretable.
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"done -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
