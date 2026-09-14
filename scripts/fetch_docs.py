"""Fetch the source documentation into data/raw/.

Shallow clones only. The docs are not vendored into this repository because they
belong to the exchanges and change often; re-run this to refresh the corpus.
"""

import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import RAW_DIR

SOURCES = {
    "binance-spot": "https://github.com/binance/binance-spot-api-docs.git",
    "bybit": "https://github.com/bybit-exchange/docs.git",
}


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

    print(f"\ndone -> {RAW_DIR}")


if __name__ == "__main__":
    main()
