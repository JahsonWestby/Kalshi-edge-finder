import requests
import time
import json
from pathlib import Path
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from logic.normalize import normalize_team, normalize_player, normalize_nba_team
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config.settings import (
    ODDS_API_KEY,
    ODDS_SPORTS,
    ODDS_BOOKMAKERS,
    ODDS_REGIONS,
    ODDS_CACHE_TTL_SEC,
    ODDS_CACHE_LOG,
    AUTO_DETECT_TENNIS,
)
import re

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORTS_API_URL = "https://api.the-odds-api.com/v4/sports"
CACHE_PATH = Path("data/odds_cache.json")
ODDS_TIMEOUT = (5, 20)
TENNIS_SPORTS_CACHE_TTL_SEC = 4 * 3600  # re-fetch active tennis list every 4 hours

_tennis_cache: list[str] = []
_tennis_cache_ts: float = 0.0


def _get_active_tennis_sports() -> list[str]:
    """Fetch active ATP/WTA tournament keys from the Odds API /v4/sports endpoint.
    Results are cached in-process for 4 hours."""
    global _tennis_cache, _tennis_cache_ts
    if _tennis_cache and (time.time() - _tennis_cache_ts) < TENNIS_SPORTS_CACHE_TTL_SEC:
        return _tennis_cache
    if not ODDS_API_KEY:
        return _tennis_cache
    try:
        r = SESSION.get(SPORTS_API_URL, params={"apiKey": ODDS_API_KEY}, timeout=ODDS_TIMEOUT)
        r.raise_for_status()
        sports = r.json()
        tennis = [
            s["key"] for s in sports
            if s.get("key", "").startswith(("tennis_atp_", "tennis_wta_"))
            and not s.get("has_outrights", False)
        ]
        _tennis_cache = tennis
        _tennis_cache_ts = time.time()
        print(f"[INFO] Active tennis sports detected: {tennis}")
        return tennis
    except Exception as exc:
        print(f"[WARN] Failed to fetch active tennis sports: {exc}")
        return _tennis_cache  # return last known list on error


def _resolve_sports() -> list[str]:
    """Return the effective sports list: ODDS_SPORTS (non-tennis) + auto-detected tennis."""
    if not AUTO_DETECT_TENNIS:
        return ODDS_SPORTS
    base = [s for s in ODDS_SPORTS if not s.startswith("tennis_")]
    return base + _get_active_tennis_sports()

_odds_retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=_odds_retry))
SESSION.mount("http://", HTTPAdapter(max_retries=_odds_retry))


def _cache_key(markets: str, sports: list[str] | None = None) -> str:
    return json.dumps(
        {
            "markets": markets,
            "regions": ODDS_REGIONS,
            "bookmakers": ODDS_BOOKMAKERS,
            "sports": sports if sports is not None else ODDS_SPORTS,
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


# ---------------------------------------------------------------------------
# Tiered book anchoring
#   Tier 1 anchor : Pinnacle (sharpest, always preferred)
#   Tier 1 fallback: LowVig + BetOnlineAG (average when Pinnacle absent)
#   Signal books  : Betfair exchanges (divergence detection only)
# ---------------------------------------------------------------------------
_PINNACLE = "pinnacle"
_TIER1_FALLBACK = ("lowvig", "betonlineag")
_BETFAIR_KEYS = ("betfair_ex_eu", "betfair_ex_uk")


def _anchor_odds(odds_by_book: dict, team: str) -> float | None:
    """Return Pinnacle odds if available, else average of tier-1 fallbacks."""
    if _PINNACLE in odds_by_book and team in odds_by_book[_PINNACLE]:
        return odds_by_book[_PINNACLE][team]
    fallback = [
        odds_by_book[b][team]
        for b in _TIER1_FALLBACK
        if b in odds_by_book and team in odds_by_book[b]
    ]
    return _avg(fallback)


def _anchor_book_label(odds_by_book: dict, team: str) -> str:
    if _PINNACLE in odds_by_book and team in odds_by_book[_PINNACLE]:
        return "pinnacle"
    used = [b for b in _TIER1_FALLBACK if b in odds_by_book and team in odds_by_book[b]]
    return f"avg({','.join(used)})" if used else "none"


def _betfair_team_odds(odds_by_book: dict, team: str) -> float | None:
    for key in _BETFAIR_KEYS:
        if key in odds_by_book and team in odds_by_book[key]:
            return odds_by_book[key][team]
    return None


def _anchor_totals(by_book: dict, side: str) -> float | None:
    """Tiered anchor for a single totals side (over/under)."""
    if _PINNACLE in by_book and side in by_book[_PINNACLE]:
        return by_book[_PINNACLE][side]
    fallback = [
        by_book[b][side]
        for b in _TIER1_FALLBACK
        if b in by_book and side in by_book[b]
    ]
    return _avg(fallback)


def _betfair_totals_odds(by_book: dict, side: str) -> float | None:
    for key in _BETFAIR_KEYS:
        if key in by_book and side in by_book[key]:
            return by_book[key][side]
    return None


def _select_books(
    books: list[dict],
    allowed_books: list[str],
    preferred_keys: list[str] | None = None,
    require_preferred: bool = False,
) -> list[dict]:
    if allowed_books:
        books = [b for b in books if b.get("key") in allowed_books]
    if preferred_keys:
        preferred = [b for b in books if b.get("key") in preferred_keys]
        if preferred:
            return preferred
        if require_preferred:
            return []
    return books


def _fetch_odds_data(markets: str):
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")

    sports = _resolve_sports()
    cache_key = _cache_key(markets, sports)

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
    for sport in sports:
        url = ODDS_API_URL.format(sport=sport)
        try:
            r = SESSION.get(url, params=params, timeout=ODDS_TIMEOUT)
            if r.status_code == 429:
                time.sleep(1.5)
                r = SESSION.get(url, params=params, timeout=ODDS_TIMEOUT)
            if r.status_code == 404:
                # Sport not supported by this API key/provider
                continue
            r.raise_for_status()
            data.extend(r.json())
        except requests.exceptions.RequestException as exc:
            print(f"[WARN] Odds API request failed for {sport}: {exc}")
            continue

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

    if not data and CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
            entry = cache.get(cache_key)
            if entry and entry.get("data"):
                print("[WARN] Odds API empty response; using cached data.")
                return entry.get("data", [])
        except Exception:
            pass
    return data


def get_moneyline_games(
    target_date: date | None = None,
    tz_name: str = "US/Central",
    date_window_days: int = 0,
):
    """
    Fetch moneyline odds from the Odds API for a target local date.

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
    data = _fetch_odds_data(markets="h2h,totals")

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
        if not books:
            continue

        sport_key = game.get("sport_key") or ""
        _is_tennis = sport_key.startswith("tennis_")
        _is_nba = sport_key == "basketball_nba"

        def _norm_name(raw: str) -> str:
            if _is_tennis:
                return normalize_player(raw)
            if _is_nba:
                return normalize_nba_team(raw)
            return normalize_team(raw)

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

                    team = _norm_name(raw_name)

                    price = outcome.get("price")
                    if price is None:
                        continue
                    odds_by_team.setdefault(team, []).append(price)
                    book_odds[team] = price
            if book_odds:
                odds_by_book[book_key] = book_odds

        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = _norm_name(home_raw)
        away = _norm_name(away_raw)

        # Tiered anchoring: Pinnacle first, then average of tier-1 fallbacks
        odds_home = _anchor_odds(odds_by_book, home)
        odds_away = _anchor_odds(odds_by_book, away)

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
                    "anchor_book": _anchor_book_label(odds_by_book, home),
                    "betfair_odds_home": _betfair_team_odds(odds_by_book, home),
                    "betfair_odds_away": _betfair_team_odds(odds_by_book, away),
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
    data = _fetch_odds_data(markets="h2h,totals")

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
        if not books:
            continue

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
        totals_by_book = totals_by_line_by_book.get(best_line, {})
        # Tiered anchoring: Pinnacle first, then average of tier-1 fallbacks
        over_avg = _anchor_totals(totals_by_book, "over")
        under_avg = _anchor_totals(totals_by_book, "under")
        if over_avg is None or under_avg is None:
            continue

        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        sport_key = game.get("sport_key") or ""
        if sport_key.startswith("tennis_"):
            home = normalize_player(home_raw)
            away = normalize_player(away_raw)
        elif sport_key == "basketball_nba":
            home = normalize_nba_team(home_raw)
            away = normalize_nba_team(away_raw)
        else:
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
                    "anchor_book": _anchor_book_label(totals_by_book, "over") if totals_by_book else "none",
                    "betfair_over_odds": _betfair_totals_odds(totals_by_book, "over"),
                    "betfair_under_odds": _betfair_totals_odds(totals_by_book, "under"),
                    "totals_by_book": totals_by_book,
                    "commence_time": game.get("commence_time"),
                    "sport_key": game.get("sport_key"),
                }
            )

    return games
