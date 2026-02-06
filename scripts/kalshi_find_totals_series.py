import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from APIs.kalshi_api import BASE_URL, TIMEOUT, kalshi_headers
import requests


KEYWORDS = [
    "total",
    "totals",
    "over",
    "under",
    "1h",
    "2h",
    "half",
    "first half",
    "second half",
    "team total",
]


def fetch_series(category: str | None = "sports", limit: int = 1000, max_pages: int = 10):
    series = []
    cursor = None
    for _ in range(max_pages):
        params = {"limit": limit}
        if category:
            params["category"] = category
        if cursor:
            params["cursor"] = cursor
        path = "/trade-api/v2/series"
        url = BASE_URL + path
        headers = kalshi_headers("GET", path)
        r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"[WARN] {r.status_code} for {url} params={params}")
            print(r.text[:1000])
            r.raise_for_status()
        data = r.json()
        if not series:
            print("[DEBUG] Response keys:", list(data.keys()))
        batch = data.get("series") or data.get("data") or []
        series.extend(batch)
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break
    return series


def is_totals_series(s):
    title = (s.get("title") or "").lower()
    subtitle = (s.get("sub_title") or "").lower()
    tags = " ".join([str(t).lower() for t in (s.get("tags") or [])])
    haystack = " ".join([title, subtitle, tags])
    return any(k in haystack for k in KEYWORDS)


def main():
    print("[INFO] Pulling sports series...")
    series = fetch_series(category="sports")
    print(f"[INFO] Total sports series: {len(series)}")

    totals = [s for s in series if is_totals_series(s)]
    print(f"[INFO] Totals-like series found: {len(totals)}")
    for s in totals:
        print(
            "-",
            s.get("ticker"),
            "| title:",
            s.get("title"),
            "| sub_title:",
            s.get("sub_title"),
            "| tags:",
            s.get("tags"),
        )

    if not series:
        print("[INFO] No sports series returned. Trying without category filter...")
        all_series = fetch_series(category=None)
        print(f"[INFO] Total series (no category): {len(all_series)}")
        totals = [s for s in all_series if is_totals_series(s)]
        print(f"[INFO] Totals-like series (no category): {len(totals)}")
        for s in totals:
            print(
                "-",
                s.get("ticker"),
                "| title:",
                s.get("title"),
                "| sub_title:",
                s.get("sub_title"),
                "| tags:",
                s.get("tags"),
            )


if __name__ == "__main__":
    main()
