import os
import sys
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ODDS_API_KEY

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def main():
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")

    sport = "basketball_ncaab"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }

    url = ODDS_API_URL.format(sport=sport)
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    books = set()
    for game in data:
        for book in game.get("bookmakers", []):
            if book.get("key"):
                books.add(book["key"])

    print(f"[INFO] Bookmakers for {sport}: {sorted(books)}")


if __name__ == "__main__":
    main()
