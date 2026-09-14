"""Ask a question from the terminal.

    python ask.py "how do I cancel an order on Bybit"
"""

import sys

from rag import answer


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python ask.py "your question"')

    result = answer(" ".join(sys.argv[1:]))

    print(result["answer"])
    print(f"\nsources  (answered by {result['provider']})")
    for source in result["sources"]:
        print(f"  [{source['n']}] {source['exchange']}  {source['source']}")


if __name__ == "__main__":
    main()
