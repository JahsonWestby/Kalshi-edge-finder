import requests
import time
import json
from pathlib import Path
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from logic.normalize import normalize_team
from config.settings import (
    ODDS_API_KEY,
    ODDS_SPORTS,
    ODDS_BOOKMAKERS,
    ODDS_REGIONS,
    ODDS_CACHE_TTL_SEC,
    ODDS_CACHE_LOG,
)
import re

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
CACHE_PATH = Path("data/odds_cache.json")


def _cache_key(markets: str) -> str:
    return json.dumps(
        {
            "markets": markets,
            "regions": ODDS_REGIONS,
            "bookmakers": ODDS_BOOKMAKERS,
            "sports": ODDS_SPORTS,
        },
        sort_keys=True,
    )


def odds_cache_remaining(markets: str) -> int | None:
    if not ODDS_CACHE_TTL_SEC or ODDS_CACHE_TTL_SEC <= 0:
        return None
    if not CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(CACHE_PATH.read_text())
    except Exception:
        return None
    entry = cache.get(_cache_key(markets))
    if not entry:
        return None
    ts = entry.get("ts", 0)
    remaining = int(ODDS_CACHE_TTL_SEC - (time.time() - ts))
    return max(0, remaining)

def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fetch_odds_data(markets: str):
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")

    cache_key = _cache_key(markets)

    if ODDS_CACHE_TTL_SEC and ODDS_CACHE_TTL_SEC > 0 and CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}
        entry = cache.get(cache_key)
        if entry:
            ts = entry.get("ts", 0)
            if (time.time() - ts) < ODDS_CACHE_TTL_SEC:
                if ODDS_CACHE_LOG:
                    print(f"[INFO] Odds API cache hit: {markets}")
                return entry.get("data", [])

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": markets,
        "oddsFormat": "american",
    }
    if ODDS_BOOKMAKERS:
        params["bookmakers"] = ODDS_BOOKMAKERS

    data = []
    for sport in ODDS_SPORTS:
        url = ODDS_API_URL.format(sport=sport)
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(1.5)
            r = requests.get(url, params=params, timeout=10)
        if r.status_code == 404:
            # Sport not supported by this API key/provider
            continue
        r.raise_for_status()
        data.extend(r.json())

    if ODDS_CACHE_TTL_SEC and ODDS_CACHE_TTL_SEC > 0:
        try:
            cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        except Exception:
            cache = {}
        cache[cache_key] = {"ts": time.time(), "data": data}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if ODDS_CACHE_LOG:
            print(f"[INFO] Odds API cache refresh: {markets}")

    return data


def get_moneyline_games(
    target_date: date | None = None,
    tz_name: str = "US/Central",
    date_window_days: int = 0,
):
    """
    Fetch NCAAB moneyline odds from Pinnacle for a target local date.

    Returns:
    [
        {
            "home": "team name",
            "away": "team name",
            "odds_home": int,
            "odds_away": int,
            "commence_time": str,  # ISO 8601 UTC
        }
    ]
    """
    data = _fetch_odds_data(markets="h2h")

    games = []
    seen = set()
    now = datetime.now(timezone.utc)
    local_tz = ZoneInfo(tz_name)
    target_date = target_date or now.astimezone(local_tz).date()
    end_date = target_date.fromordinal(target_date.toordinal() + date_window_days)

    allowed_books = [b.strip() for b in (ODDS_BOOKMAKERS or "").split(",") if b.strip()]

    for game in data:
        start_time = datetime.fromisoformat(
            game["commence_time"].replace("Z", "+00:00")
        )

        # Filter to a specific local calendar date
        local_date = start_time.astimezone(local_tz).date()
        if not (target_date <= local_date <= end_date):
            continue

        odds_by_team = {}
        odds_by_book = {}
        books = game.get("bookmakers", [])
        if allowed_books:
            books = [b for b in books if b.get("key") in allowed_books]
        else:
            # Prefer Pinnacle if present, otherwise fall back to first book.
            books = [b for b in books if b.get("key") == "pinnacle"] or books[:1]

        for book in books:
            book_key = book.get("key") or "book"
            book_odds = {}
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    raw_name = outcome.get("name", "")

                    # remove parentheticals like "(Chi)"
                    raw_name = re.sub(r"\(.*?\)", "", raw_name).strip()

                    team = normalize_team(raw_name)

                    price = outcome.get("price")
                    if price is None:
                        continue
                    odds_by_team.setdefault(team, []).append(price)
                    book_odds[team] = price
            if book_odds:
                odds_by_book[book_key] = book_odds

        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)
        odds_home = _avg(odds_by_team.get(home, []))
        odds_away = _avg(odds_by_team.get(away, []))

        if home and away and odds_home is not None and odds_away is not None:
            key = (away, home, game.get("commence_time"))
            if key in seen:
                continue
            seen.add(key)
            games.append(
                {
                    "home": home,
                    "away": away,
                    "odds_home": odds_home,
                    "odds_away": odds_away,
                    "odds_by_book": odds_by_book,
                    "commence_time": game.get("commence_time"),
                    "sport_key": game.get("sport_key"),
                }
            )

    return games


def get_moneyline_odds(target_date: date | None = None, tz_name: str = "US/Central"):
    """
    Backwards-compatible: returns a flat mapping of team -> odds.
    """
    odds = {}
    for game in get_moneyline_games(target_date=target_date, tz_name=tz_name):
        odds[game["home"]] = game["odds_home"]
        odds[game["away"]] = game["odds_away"]
    return odds


def get_totals_games(
    target_date: date | None = None,
    tz_name: str = "US/Central",
    date_window_days: int = 0,
):
    """
    Fetch NCAAB game totals (over/under) odds from Pinnacle for a target local date.

    Returns:
    [
        {
            "home": "team name",
            "away": "team name",
            "total": float,
            "over_odds": int,
            "under_odds": int,
            "commence_time": str,  # ISO 8601 UTC
        }
    ]
    """
    data = _fetch_odds_data(markets="totals")

    games = []
    seen = set()
    now = datetime.now(timezone.utc)
    local_tz = ZoneInfo(tz_name)
    target_date = target_date or now.astimezone(local_tz).date()
    end_date = target_date.fromordinal(target_date.toordinal() + date_window_days)

    allowed_books = [b.strip() for b in (ODDS_BOOKMAKERS or "").split(",") if b.strip()]

    for game in data:
        start_time = datetime.fromisoformat(
            game["commence_time"].replace("Z", "+00:00")
        )

        # Filter to a specific local calendar date
        local_date = start_time.astimezone(local_tz).date()
        if not (target_date <= local_date <= end_date):
            continue

        totals_by_line = {}
        totals_by_line_by_book = {}
        books = game.get("bookmakers", [])
        if allowed_books:
            books = [b for b in books if b.get("key") in allowed_books]
        else:
            books = [b for b in books if b.get("key") == "pinnacle"] or books[:1]

        for book in books:
            book_key = book.get("key") or "book"
            for market in book.get("markets", []):
                if market.get("key") != "totals":
                    continue
                over = None
                under = None
                total_points = None
                for outcome in market.get("outcomes", []):
                    name = (outcome.get("name") or "").lower()
                    point = outcome.get("point")
                    if point is not None:
                        total_points = float(point)
                    if name == "over":
                        over = outcome.get("price")
                    elif name == "under":
                        under = outcome.get("price")
                if total_points is not None and over is not None and under is not None:
                    totals_by_line.setdefault(total_points, {"over": [], "under": []})
                    totals_by_line[total_points]["over"].append(over)
                    totals_by_line[total_points]["under"].append(under)
                    totals_by_line_by_book.setdefault(total_points, {})
                    totals_by_line_by_book[total_points][book_key] = {
                        "over": over,
                        "under": under,
                    }

        if not totals_by_line:
            continue
        # Choose the line with the most data points (both sides)
        best_line = None
        best_count = -1
        for line, sides in totals_by_line.items():
            count = min(len(sides["over"]), len(sides["under"]))
            if count > best_count:
                best_count = count
                best_line = line
        if best_line is None:
            continue
        over_avg = _avg(totals_by_line[best_line]["over"])
        under_avg = _avg(totals_by_line[best_line]["under"])
        if over_avg is None or under_avg is None:
            continue
        totals_by_book = totals_by_line_by_book.get(best_line, {})

        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)
        if home and away:
            key = (away, home, best_line, game.get("commence_time"))
            if key in seen:
                continue
            seen.add(key)
            games.append(
                {
                    "home": home,
                    "away": away,
                    "total": best_line,
                    "over_odds": over_avg,
                    "under_odds": under_avg,
                    "totals_by_book": totals_by_book,
                    "commence_time": game.get("commence_time"),
                    "sport_key": game.get("sport_key"),
                }
            )

    return games
