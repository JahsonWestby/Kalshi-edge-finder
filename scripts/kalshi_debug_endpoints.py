import json
import os
import sys
import requests

# Ensure repo root is on path when running from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from APIs.kalshi_api import BASE_URL, TIMEOUT, kalshi_headers


ENDPOINTS = [
    "/trade-api/v2/portfolio/positions",
    "/trade-api/v2/portfolio/orders",
    "/trade-api/v2/portfolio/fills",
    "/trade-api/v2/portfolio/balance",
]


def fetch(path: str):
    url = BASE_URL + path
    headers = kalshi_headers("GET", path)
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    except Exception as exc:
        return {"error": str(exc)}

    out = {"status": resp.status_code}
    try:
        data = resp.json()
        out["json_keys"] = list(data.keys()) if isinstance(data, dict) else None
        out["sample"] = data if isinstance(data, dict) else data[:2]
    except Exception:
        out["text"] = resp.text[:500]
    return out


def main():
    results = {}
    for path in ENDPOINTS:
        results[path] = fetch(path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
