import os
import sys
import requests
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ODDS_API_KEY

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def main():
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")

    sport = "basketball_ncaab"
    url = ODDS_API_URL.format(sport=sport)

    candidate_markets = [
        "h2h",
        "totals",
        "team_totals",
        "spreads",
        "alternate_totals",
        "alternate_team_totals",
        "h2h_1h",
        "totals_1h",
        "team_totals_1h",
        "h2h_2h",
        "totals_2h",
        "team_totals_2h",
    ]

    supported = []
    for key in candidate_markets:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": key,
            "oddsFormat": "american",
            "bookmakers": "pinnacle",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            print("[WARN] 429 rate limit hit. Stop probing; wait a minute and rerun.")
            break
        if r.status_code == 404:
            print(f"[WARN] 404 for market={key}")
            continue
        if r.status_code == 422:
            print(f"[WARN] 422 unsupported market={key}")
            continue
        r.raise_for_status()
        data = r.json()
        if data:
            supported.append(key)
        else:
            print(f"[INFO] market={key} returned 0 events")
        time.sleep(0.5)

    print("[INFO] Supported markets:", supported)

    if not supported:
        return

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(supported),
        "oddsFormat": "american",
        "bookmakers": "pinnacle",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    market_keys = set()
    bookmaker_keys = set()
    for game in data:
        for book in game.get("bookmakers", []):
            bookmaker_keys.add(book.get("key"))
            for market in book.get("markets", []):
                market_keys.add(market.get("key"))

    print("[INFO] Bookmakers found:", sorted(k for k in bookmaker_keys if k))
    print("[INFO] Market keys found:", sorted(k for k in market_keys if k))

    # Print a small sample of totals/team_totals structures for inspection
    for game in data[:3]:
        print("\n[INFO] Sample game:", game.get("home_team"), "vs", game.get("away_team"))
        for book in game.get("bookmakers", []):
            if book.get("key") != "pinnacle":
                continue
            for market in book.get("markets", []):
                if market.get("key") in ("totals", "team_totals", "alternate_totals", "alternate_team_totals"):
                    print(" - market:", market.get("key"))
                    for outcome in market.get("outcomes", [])[:6]:
                        print("   ", outcome)


if __name__ == "__main__":
    main()
