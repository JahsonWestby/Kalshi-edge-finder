import os
import sys
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ODDS_API_KEY

SPORTS_URL = "https://api.the-odds-api.com/v4/sports"


def main():
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")
    params = {"apiKey": ODDS_API_KEY}
    r = requests.get(SPORTS_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    print(f"[INFO] Total sports: {len(data)}")
    for s in data:
        print("-", s.get("key"), "|", s.get("title"))


if __name__ == "__main__":
    main()
