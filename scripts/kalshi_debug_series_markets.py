import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from APIs.kalshi_api import get_all_markets


def main():
    series_ticker = "KXNCAAMBTOTAL"
    markets = get_all_markets(params={"series_ticker": series_ticker}, max_pages=3)
    print(f"[INFO] Markets for {series_ticker}: {len(markets)}")
    if not markets:
        return
    print("[INFO] Sample market keys:", list(markets[0].keys()))
    for m in markets[:10]:
        print(
            "-",
            m.get("ticker"),
            "| title:",
            m.get("title"),
            "| subtitle:",
            m.get("subtitle") or m.get("sub_title") or m.get("yes_sub_title"),
            "| status:",
            m.get("status"),
        )


if __name__ == "__main__":
    main()
