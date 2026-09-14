"""Ingestion: which files are indexed, and what is left of them afterwards.

Most of these cases are regressions. Ingestion fails quietly by nature - a page
that is skipped, or stripped down to nothing, produces no error and no warning,
only a slightly worse answer weeks later.
"""

from pathlib import Path

import pytest

import ingest


def write(root: Path, relative: str, text: str = "x" * 200) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A miniature data/raw with one example of every exclusion rule."""
    write(tmp_path, "binance-spot/rest-api.md")
    write(tmp_path, "bybit/docs/v5/order/create-order.mdx")
    write(tmp_path, "binance-spot/rest-api_CN.md")
    write(tmp_path, "binance-spot/CHANGELOG.md")
    write(tmp_path, "bybit/docs/v5/faq.mdx")
    write(tmp_path, "bybit/i18n/zh/order.mdx")
    write(tmp_path, "bybit/docs/api-explorer/order.mdx")
    write(tmp_path, "binance-spot/testnet/rest-api.md")
    write(tmp_path, "bybit/docs/v5/demo-mode/order.mdx")
    write(tmp_path, "bybit/changelog/2024.mdx")
    write(tmp_path, "bybit/.git/COMMIT_EDITMSG")
    write(tmp_path, "bybit/package.json")
    write(tmp_path, "bybit/static/logo.png")
    return tmp_path


def relative_names(root: Path) -> set[str]:
    return {
        str(path.relative_to(root)).replace("\\", "/")
        for path in ingest.collect_files(root)
    }


def test_collects_both_markdown_suffixes(corpus: Path) -> None:
    """Bybit ships .mdx. Matching only .md indexed 3% of that corpus in silence."""
    found = relative_names(corpus)
    assert "binance-spot/rest-api.md" in found
    assert "bybit/docs/v5/order/create-order.mdx" in found


@pytest.mark.parametrize(
    "excluded",
    [
        "binance-spot/rest-api_CN.md",
        "binance-spot/CHANGELOG.md",
        "bybit/docs/v5/faq.mdx",
        "bybit/i18n/zh/order.mdx",
        "bybit/docs/api-explorer/order.mdx",
        "bybit/docs/v5/demo-mode/order.mdx",
        "bybit/changelog/2024.mdx",
        "bybit/.git/COMMIT_EDITMSG",
        "bybit/package.json",
        "bybit/static/logo.png",
    ],
)
def test_excluded_paths_are_not_collected(corpus: Path, excluded: str) -> None:
    assert excluded not in relative_names(corpus)


def test_testnet_mirror_is_excluded(corpus: Path) -> None:
    """The production page must survive and the sandbox copy must not.

    Deduplication alone kept whichever path sorted first, which is the testnet
    one, so the sandbox rate limits were indexed and the production limits were
    dropped. The exclusion is what makes the surviving copy the right one.
    """
    found = relative_names(corpus)
    assert "binance-spot/rest-api.md" in found
    assert "binance-spot/testnet/rest-api.md" not in found


def test_clean_extracts_frontmatter_title_and_removes_the_block() -> None:
    body, title = ingest.clean(
        "---\ntitle: Place Order\nsidebar_label: Order\n---\n\nPlace a new order.\n"
    )
    assert title == "Place Order"
    assert body == "Place a new order."


def test_clean_accepts_crlf_frontmatter() -> None:
    """The corpus is cloned on Windows too, where git may check out CRLF."""
    body, title = ingest.clean("---\r\ntitle: Amend Order\r\n---\r\nAmend it.\r\n")
    assert title == "Amend Order"
    assert "---" not in body


def test_clean_without_frontmatter_returns_no_title() -> None:
    body, title = ingest.clean("# Cancel Order\n\nCancels an order.")
    assert title == ""
    assert body.startswith("# Cancel Order")


def test_clean_strips_docusaurus_scaffolding_but_keeps_the_prose() -> None:
    body, _ = ingest.clean(
        'import Tabs from "@theme/Tabs";\n'
        "\n"
        ":::info\n"
        "Rate limited to 10 requests per second.\n"
        ":::\n"
        "\n"
        "<Tabs>Supported order types</Tabs>\n"
    )
    assert "import" not in body
    assert ":::" not in body
    assert "<Tabs>" not in body
    assert "Rate limited to 10 requests per second." in body
    assert "Supported order types" in body


def test_clean_collapses_blank_runs() -> None:
    body, _ = ingest.clean("First.\n\n\n\n\nSecond.")
    assert body == "First.\n\nSecond."


def test_split_file_records_source_exchange_and_section(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "bybit/docs/v5/order/create-order.mdx",
        "# Create Order\n\n## Request Parameters\n\n" + "detail. " * 40,
    )
    chunks = ingest.split_file(path, tmp_path)

    assert chunks
    meta = chunks[0].metadata
    assert meta["source"] == "bybit/docs/v5/order/create-order.mdx"
    assert meta["exchange"] == "bybit"
    assert meta["section"] == "Create Order > Request Parameters"


def test_split_file_uses_forward_slashes_on_every_platform(tmp_path: Path) -> None:
    """source is compared against questions.json, which is written with '/'."""
    path = write(tmp_path, "binance-spot/rest-api.md", "# Rest\n\n" + "body. " * 40)
    assert "\\" not in ingest.split_file(path, tmp_path)[0].metadata["source"]


def test_split_file_promotes_the_frontmatter_title_to_a_heading(
    tmp_path: Path,
) -> None:
    """Without this the topic appears nowhere in the chunk that should carry it."""
    path = write(
        tmp_path,
        "bybit/docs/v5/position/leverage.mdx",
        "---\ntitle: Set Leverage\n---\n\n" + "Adjust the leverage. " * 10,
    )
    chunks = ingest.split_file(path, tmp_path)

    assert "Set Leverage" in chunks[0].page_content
    assert chunks[0].metadata["section"] == "Set Leverage"


def test_split_file_skips_a_page_with_no_content(tmp_path: Path) -> None:
    path = write(tmp_path, "bybit/docs/v5/stub.mdx", "---\ntitle: Stub\n---\n")
    assert ingest.split_file(path, tmp_path) == []


def test_deduplicate_keeps_the_first_copy_and_preserves_order() -> None:
    from langchain_core.documents import Document

    documents = [
        Document(page_content="alpha", metadata={"source": "a.md"}),
        Document(page_content="beta", metadata={"source": "b.md"}),
        Document(page_content="alpha", metadata={"source": "testnet/a.md"}),
    ]
    unique = ingest.deduplicate(documents)

    assert [doc.page_content for doc in unique] == ["alpha", "beta"]
    assert unique[0].metadata["source"] == "a.md"
