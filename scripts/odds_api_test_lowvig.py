import os
import sys
import requests
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ODDS_API_KEY, ODDS_REGIONS

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def _get(url, params):
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 429:
        time.sleep(1.5)
        r = requests.get(url, params=params, timeout=10)
    return r


def _print_sample(data, label):
    if not data:
        print(f"[INFO] {label}: 0 games")
        return
    game = data[0]
    home = game.get("home_team")
    away = game.get("away_team")
    books = [b.get("key") for b in game.get("bookmakers", []) if b.get("key")]
    print(f"[INFO] {label}: {len(data)} games | sample: {away} at {home} | books: {books[:5]}")


def main():
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")

    sport = "basketball_ncaab"
    regions = ODDS_REGIONS or ""
    base_params = {
        "apiKey": ODDS_API_KEY,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    if regions:
        base_params["regions"] = regions

    url = ODDS_API_URL.format(sport=sport)

    # First: no bookmaker filter
    r = _get(url, params=base_params)
    if r.status_code != 200:
        print(f"[WARN] No-bookmaker call failed: {r.status_code} {r.text}")
    else:
        _print_sample(r.json(), "No-bookmaker")

    # Try LowVig keys
    lowvig_keys = ["lowvig", "lowvig_ag", "lowvigag", "lowvig.ag"]
    for key in lowvig_keys:
        params = dict(base_params)
        params["bookmakers"] = key
        r = _get(url, params=params)
        if r.status_code != 200:
            print(f"[WARN] key={key} -> {r.status_code}")
            continue
        data = r.json()
        _print_sample(data, f"key={key}")


if __name__ == "__main__":
    main()
