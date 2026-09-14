"""Shared access to the persisted index.

Both retrieval halves read from here: the dense side queries Chroma directly, the
keyword side needs every chunk in memory to build its own index. Keeping the
loader in one place means the two can never drift onto different corpora.
"""

from typing import TYPE_CHECKING

from langchain_core.documents import Document

from config import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL

if TYPE_CHECKING:
    from langchain_chroma import Chroma

_store: "Chroma | None" = None
_documents: list[Document] | None = None


def get_store() -> "Chroma":
    """Loading the embedding model takes a few seconds, so keep one instance.

    Chroma and the embedding backend are imported here rather than at module
    level because importing them pulls in torch, which costs seconds and several
    hundred megabytes. Nothing that only inspects retrieval logic should pay for
    that, and the import happens once on the first call either way.
    """
    global _store
    if _store is None:
        if not CHROMA_DIR.exists():
            raise SystemExit("no index - run ingest.py first")

        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return _store


def all_documents() -> list[Document]:
    """Every indexed chunk, read back out of Chroma.

    Reading the corpus back from the vector store rather than re-reading data/raw
    guarantees the keyword index sees exactly what was embedded - same cleaning,
    same chunk boundaries, same deduplication. A production system would put the
    keyword index in a real search engine instead of rebuilding it per process;
    at 5k chunks the rebuild costs under a second and is not worth the extra
    moving part.
    """
    global _documents
    if _documents is None:
        raw = get_store().get(include=["documents", "metadatas"])
        _documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(
                raw["documents"], raw["metadatas"], strict=True
            )
        ]
    return _documents
