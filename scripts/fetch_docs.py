"""Fetch the source documentation into data/raw/.

Shallow clones only. The docs are not vendored into this repository because they
belong to the exchanges and change often; re-run this to refresh the corpus.

Because they change often, the commit each corpus was fetched at is printed and
written to data/raw/CORPUS.txt. A retrieval measurement is only reproducible
against a known corpus: a page renamed upstream turns an expected answer into a
permanent miss, and without the commit there is no way to tell that apart from a
change in this repository having made retrieval worse.
"""

import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import RAW_DIR

SOURCES = {
    "binance-spot": "https://github.com/binance/binance-spot-api-docs.git",
    "bybit": "https://github.com/bybit-exchange/docs.git",
}


def commit_of(name: str) -> str:
    return subprocess.run(
        ["git", "-C", str(RAW_DIR / name), "log", "-1", "--format=%h %cs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def record_versions() -> None:
    lines = [f"{name} {commit_of(name)}" for name in SOURCES if (RAW_DIR / name).exists()]
    for line in lines:
        print(line)
    (RAW_DIR / "CORPUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in SOURCES.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"{name}: already present, skipping")
            continue

        print(f"{name}: cloning {url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            check=True,
        )

    print("\ncorpus:")
    record_versions()
    print(f"\ndone -> {RAW_DIR}")


if __name__ == "__main__":
    main()
