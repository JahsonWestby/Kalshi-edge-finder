import base64
import time
import re
from datetime import datetime, timezone, date

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from logic.normalize import normalize_team
from config.settings import (
    KALSHI_KEY_ID,
    KALSHI_PEM_PATH,
    KALSHI_PRICE_MODE,
    KALSHI_SERIES_TICKERS,
    KALSHI_TOTALS_SERIES_TICKERS,
)

BASE_URL = "https://api.elections.kalshi.com"
# NOTE: Kalshi production trading API is hosted on api.elections.kalshi.com.
# api.kalshi.com does not resolve for trade-api endpoints and will cause DNS errors.
TIMEOUT = 20

if not KALSHI_KEY_ID:
    raise RuntimeError("KALSHI_KEY_ID is not set")
KALSHI_PRIVATE_KEY_PATH = str(KALSHI_PEM_PATH)

with open(KALSHI_PRIVATE_KEY_PATH, "rb") as f:
    PRIVATE_KEY = serialization.load_pem_private_key(
        f.read(),
        password=None,
    )

def kalshi_headers(method: str, path: str):
    ts = str(int(time.time() * 1000))
    msg = ts + method + path

    signature = PRIVATE_KEY.sign(
        msg.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


def _fetch_paginated(path: str, list_key: str, limit=200, max_pages=5, params=None):
    items = []
    cursor = None
    params = dict(params or {})

    for _ in range(max_pages):
        page_params = {"limit": limit, **params}
        if cursor:
            page_params["cursor"] = cursor

        url = BASE_URL + path
        headers = kalshi_headers("GET", path)

        r = requests.get(
            url,
            params=page_params,
            headers=headers,
            timeout=TIMEOUT,
        )
        r.raise_for_status()

        data = r.json()
        batch = data.get(list_key) or data.get("data") or []
        items.extend(batch)

        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break

    return items


def get_all_events(limit=200, max_pages=5, params=None):
    return _fetch_paginated(
        path="/trade-api/v2/events",
        list_key="events",
        limit=limit,
        max_pages=max_pages,
        params=params,
    )


def _extract_series_title(series: dict) -> str:
    for key in (
        "title",
        "name",
        "series_title",
        "event_title",
        "series_name",
        "event_name",
    ):
        value = series.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_sports_series(series: dict) -> bool:
    category = (series.get("category") or series.get("type") or "").lower()
    tags = [
        t.lower()
        for t in (series.get("tags") or [])
        if isinstance(t, str)
    ]
    if "sports" in category or "sport" in category:
        return True
    return "sports" in tags or "sport" in tags


def _is_ncaa_series(series: dict) -> bool:
    title = _extract_series_title(series).lower()
    ticker = (series.get("ticker") or "").lower()
    tags = [
        t.lower()
        for t in (series.get("tags") or [])
        if isinstance(t, str)
    ]
    if "college basketball" in title or "ncaa" in title or "ncaab" in title:
        return True
    if "ncaab" in ticker or "ncaa" in ticker:
        return True
    return any(key in " ".join(tags) for key in ("ncaa", "ncaab", "college"))


def _is_mens_ncaa_series(series: dict) -> bool:
    title = _extract_series_title(series).lower()
    ticker = (series.get("ticker") or "").lower()
    tags = [
        t.lower()
        for t in (series.get("tags") or [])
        if isinstance(t, str)
    ]
    if "men's" in title or "mens" in title:
        if "college basketball" in title or "ncaa" in title or "ncaab" in title:
            return True
    if "ncaam" in ticker or "ncaam" in " ".join(tags):
        return True
    return False


def get_all_series(limit=200, max_pages=5):
    return _fetch_paginated(
        path="/trade-api/v2/series",
        list_key="series",
        limit=limit,
        max_pages=max_pages,
    )


def get_all_markets(limit=200, max_pages=5, params=None):
    return _fetch_paginated(
        path="/trade-api/v2/markets",
        list_key="markets",
        limit=limit,
        max_pages=max_pages,
        params=params,
    )


def place_order(payload: dict):
    """
    Place a Kalshi order. Payload should match /trade-api/v2/portfolio/orders.
    """
    path = "/trade-api/v2/portfolio/orders"
    url = BASE_URL + path
    headers = kalshi_headers("POST", path)
    r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"Order failed {r.status_code}: {r.text}")
    return r.json()


def get_balance():
    """
    Fetch available balance and portfolio value (both in cents).
    """
    path = "/trade-api/v2/portfolio/balance"
    url = BASE_URL + path
    headers = kalshi_headers("GET", path)
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_positions():
    """
    Fetch current positions.
    """
    path = "/trade-api/v2/portfolio/positions"
    url = BASE_URL + path
    headers = kalshi_headers("GET", path)
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_orders(params=None, max_pages=10, limit=200):
    """
    Fetch orders with pagination.
    """
    path = "/trade-api/v2/portfolio/orders"
    headers = kalshi_headers("GET", path)
    all_orders = []
    cursor = None
    params = dict(params or {})

    for _ in range(max_pages):
        page_params = {"limit": limit, **params}
        if cursor:
            page_params["cursor"] = cursor
        url = BASE_URL + path
        r = requests.get(url, headers=headers, params=page_params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        batch = data.get("orders") or data.get("data") or []
        all_orders.extend(batch)
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break

    return {"orders": all_orders}


def get_trades(params=None, max_pages=10, limit=200):
    """
    Fetch trades/fills with pagination.
    """
    def _fetch(path: str, list_key: str):
        headers = kalshi_headers("GET", path)
        all_items = []
        cursor = None
        base_params = dict(params or {})
        for _ in range(max_pages):
            page_params = {"limit": limit, **base_params}
            if cursor:
                page_params["cursor"] = cursor
            url = BASE_URL + path
            r = requests.get(url, headers=headers, params=page_params, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            batch = data.get(list_key) or data.get("data") or []
            all_items.extend(batch)
            cursor = data.get("cursor") or data.get("next_cursor")
            if not cursor:
                break
        return all_items

    # Prefer fills endpoint; fallback to trades if needed.
    items = _fetch("/trade-api/v2/portfolio/fills", "fills")
    if items is None:
        items = _fetch("/trade-api/v2/portfolio/trades", "trades") or []
    return {"trades": items}


def get_market_by_ticker(ticker: str) -> dict | None:
    if not ticker:
        return None
    path = f"/trade-api/v2/markets/{ticker}"
    url = BASE_URL + path
    headers = kalshi_headers("GET", path)
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    if not r.ok:
        return None
    data = r.json()
    return data.get("market") or data


def cancel_order(order_id: str):
    """
    Cancel an existing order by order_id.
    """
    attempts = [
        ("POST", f"/trade-api/v2/portfolio/orders/{order_id}/cancel"),
        ("POST", f"/trade-api/v2/portfolio/orders/{order_id}/cancel/"),
        ("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"),
        ("POST", f"/trade-api/v2/orders/{order_id}/cancel"),
        ("DELETE", f"/trade-api/v2/orders/{order_id}"),
    ]
    last_err = None
    for method, path in attempts:
        url = BASE_URL + path
        headers = kalshi_headers(method, path)
        try:
            if method == "POST":
                r = requests.post(url, headers=headers, timeout=TIMEOUT)
            else:
                r = requests.delete(url, headers=headers, timeout=TIMEOUT)
        except Exception as exc:
            last_err = exc
            continue
        if r.ok:
            try:
                return r.json()
            except Exception:
                return {"status": r.status_code, "text": r.text}
        last_err = RuntimeError(f"Cancel failed {r.status_code}: {r.text}")
    raise RuntimeError(last_err)


def parse_balance(balance_json: dict) -> dict:
    """
    Best-effort extraction of usable balances from Kalshi response.
    Returns values in dollars.
    """
    if not isinstance(balance_json, dict):
        return {"cash_available": None, "portfolio_value": None, "raw": balance_json}

    data = balance_json.get("balance") if isinstance(balance_json, dict) else None
    if data is None:
        data = balance_json
    if isinstance(data, (int, float)):
        return {
            "cash_available": data / 100,
            "portfolio_value": None,
            "raw": balance_json,
        }
    if not isinstance(data, dict):
        return {"cash_available": None, "portfolio_value": None, "raw": balance_json}

    def _get_cents(*keys):
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        return None

    cash_cents = _get_cents(
        "cash_available",
        "available_cash",
        "cash_balance",
        "cash",
        "available_balance",
        "balance",
    )
    portfolio_cents = _get_cents(
        "portfolio_value",
        "account_value",
        "equity",
        "net_liquidation_value",
        "total_balance",
    )

    def _to_dollars(cents):
        if cents is None:
            return None
        return cents / 100

    return {
        "cash_available": _to_dollars(cash_cents),
        "portfolio_value": _to_dollars(portfolio_cents),
        "raw": balance_json,
    }


# List all sports-related series (and events) to see what Kalshi actually offers
def list_sports_series(limit=200, max_pages=5):
    print("[DEBUG] Pulling ALL series from Kalshi (client-side sports filter)...")
    all_series = get_all_series(limit=limit, max_pages=max_pages)
    sports_series = [s for s in all_series if _is_sports_series(s)]

    print(f"[DEBUG] Total sports series found: {len(sports_series)}")
    for s in sports_series:
        print(
            "-",
            s.get("ticker"),
            "| title:",
            _extract_series_title(s) or "(missing title)",
            "| tags:",
            s.get("tags"),
            "| category:",
            s.get("category"),
        )

    return sports_series


def looks_like_ncaa_game(event):
    title = (event.get("title") or "").lower()
    category = (event.get("category") or "").lower()
    tags = [
        t.lower()
        for t in (event.get("tags") or [])
        if isinstance(t, str)
    ]

    if any(
        key in category
        for key in ("ncaa", "ncaab", "college", "cbb")
    ):
        return True

    if any(
        key in " ".join(tags)
        for key in ("ncaa", "ncaab", "college", "cbb", "college-basketball")
    ):
        return True

    # Primary: basketball + vs/@ pattern
    if (" vs " in title or "@" in title) and "basketball" in category:
        return True

    # Fallback: common college matchup formatting
    if " vs " in title or "@" in title:
        # reject obvious pro leagues
        banned = ["nba", "wnba", "euroleague", "g league"]
        if any(b in title for b in banned):
            return False
        return True

    return False


def get_ncaa_events():
    print("[DEBUG] Pulling events from Kalshi...")
    events = get_all_events()
    print(f"[DEBUG] Total events fetched: {len(events)}")

    print("[DEBUG] Showing 20 sample OPEN events (title | category | tags):")
    shown = 0
    for e in events:
        status = (e.get("status") or "").lower()
        if status not in ("open", "active", "live"):
            continue

        print(
            "-",
            e.get("title"),
            "| category:",
            e.get("category"),
            "| tags:",
            e.get("tags"),
        )
        shown += 1
        if shown >= 20:
            break

    ncaa = []
    now = datetime.now(timezone.utc)

    for e in events:
        status = (e.get("status") or "").lower()
        if status not in ("open", "active", "live"):
            continue

        if not looks_like_ncaa_game(e):
            continue

        start = e.get("event_start_time") or e.get("start_time")
        if start:
            try:
                if start.endswith("Z"):
                    start = start[:-1] + "+00:00"
                dt = datetime.fromisoformat(start)
                if dt < now:
                    continue
            except Exception:
                pass

        ncaa.append(e)

    print(f"[DEBUG] NCAA-like events found: {len(ncaa)}")

    for e in ncaa[:10]:
        print("-", e.get("title"))

    return ncaa


def _parse_iso_utc(timestamp: str) -> datetime | None:
    if not timestamp:
        return None
    try:
        if timestamp.endswith("Z"):
            timestamp = timestamp[:-1] + "+00:00"
        return datetime.fromisoformat(timestamp)
    except Exception:
        return None


def _parse_ticker_date(ticker: str) -> datetime | None:
    if not ticker or "-" not in ticker:
        return None
    # Example: KXNCAAMBGAME-26FEB05CSBUCI-UCI
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    date_part = parts[1][:7]
    try:
        dt = datetime.strptime(date_part, "%y%b%d")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _format_ncaa_row(dt_local: datetime | None, event: dict) -> str:
    title = event.get("title") or "(missing title)"
    ticker = event.get("ticker") or "?"
    category = event.get("category") or "?"
    time_part = dt_local.strftime("%Y-%m-%d %H:%M %Z") if dt_local else "time-unknown"
    return f"{time_part} | {title} | {ticker} | {category}"


def _format_price(p: float | None) -> str:
    if p is None:
        return "--"
    try:
        return f"{p * 100:.1f}%"
    except Exception:
        return "--"


def _market_game_datetime(market: dict) -> tuple[datetime | None, str]:
    close_time = (
        market.get("close_time")
        or market.get("event_end_time")
        or market.get("end_time")
    )
    start = (
        market.get("event_start_time")
        or market.get("start_time")
        or close_time
    )
    if start and start != close_time:
        return _parse_iso_utc(start), "start_time"

    # Prefer the ticker-encoded date over close_time if start_time is missing.
    ticker_dt = _parse_ticker_date(market.get("ticker") or "")
    if ticker_dt:
        return ticker_dt, "ticker_date"

    if start:
        return _parse_iso_utc(start), "close_time"
    return None, "missing"


def _normalize_price(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100 if value > 1 else value


def _select_price(
    bid: float | None,
    ask: float | None,
    last: float | None,
    side: str,
) -> float | None:
    bid_n = _normalize_price(bid)
    ask_n = _normalize_price(ask)
    last_n = _normalize_price(last)

    mode = (KALSHI_PRICE_MODE or "ask").lower()
    if mode == "trade":
        if side == "yes":
            return ask_n if ask_n is not None else bid_n or last_n
        if side == "no":
            return bid_n if bid_n is not None else ask_n or last_n
    if mode == "bid":
        return bid_n if bid_n is not None else ask_n
    if mode == "mid":
        if bid_n is not None and ask_n is not None:
            return (bid_n + ask_n) / 2
        return ask_n if ask_n is not None else bid_n
    # default "ask"
    return ask_n if ask_n is not None else bid_n or last_n


def _market_yes_price(market: dict) -> float | None:
    return _select_price(
        market.get("yes_bid"),
        market.get("yes_ask"),
        market.get("last_price"),
        "yes",
    )


def _market_no_price(market: dict) -> float | None:
    return _select_price(
        market.get("no_bid"),
        market.get("no_ask"),
        None,
        "no",
    )


def _normalize_abbrev(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _team_abbrev(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name or "")
    if not words:
        return ""
    return "".join(w[0] for w in words).lower()


def _team_code_variants(name: str) -> set[str]:
    base = _normalize_abbrev(name)
    variants = set()
    if base:
        variants.add(base)
        variants.add(base[:3])
        variants.add(base[:4])
    abbrev = _team_abbrev(name)
    if abbrev:
        variants.add(abbrev)
    return {v for v in variants if v}


def _split_matchup(title: str) -> tuple[str, str] | tuple[None, None]:
    if not title:
        return (None, None)
    clean = title.replace("Winner?", "").strip()
    # Totals titles include suffixes like ": Total Points" – strip anything after a colon.
    if ":" in clean:
        clean = clean.split(":", 1)[0].strip()
    if " at " in clean:
        away, home = clean.split(" at ", 1)
        return away.strip(), home.strip()
    if " vs " in clean:
        t1, t2 = clean.split(" vs ", 1)
        return t1.strip(), t2.strip()
    return (None, None)


def get_kalshi_markets(
    target_date: date | None = None,
    tz_name: str = "US/Central",
    date_window_days: int = 14,
):
    """
    Returns a mapping of series_ticker -> {normalized team name -> market info}.
    """
    markets = []
    for ticker in KALSHI_SERIES_TICKERS:
        markets.extend(
            get_all_markets(
                params={"series_ticker": ticker},
                max_pages=10,
            )
        )

    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    target_date = target_date or now_local.date()

    result = {}
    for m in markets:
        status = (m.get("status") or "").lower()
        if status and status not in ("open", "active", "live"):
            continue

        dt_utc, dt_source = _market_game_datetime(m)
        if dt_utc:
            if dt_source == "ticker_date":
                # Date-only fallback: keep if within the window around target_date.
                start_date = target_date
                end_date = target_date.fromordinal(target_date.toordinal() + date_window_days)
                if not (start_date <= dt_utc.date() <= end_date):
                    continue
            else:
                # Use local date window for explicit timestamps.
                dt_local = dt_utc.astimezone()
                start_date = target_date
                end_date = target_date.fromordinal(target_date.toordinal() + date_window_days)
                if not (start_date <= dt_local.date() <= end_date):
                    continue

        team_key = m.get("yes_sub_title") or m.get("subtitle") or ""
        team_key = _normalize_abbrev(team_key)
        team_name = ""

        title = m.get("title") or ""
        away, home = _split_matchup(title)
        if away and home:
            away_keys = _team_code_variants(away)
            home_keys = _team_code_variants(home)
            if team_key in away_keys:
                team_name = away
            elif team_key in home_keys:
                team_name = home
            else:
                # Fallback to last token if we can't match the abbrev
                team_name = home
        else:
            team_name = team_key

        norm = normalize_team(team_name)
        if not norm:
            continue

        series = m.get("series_ticker") or ""
        if not series:
            ticker = m.get("ticker") or ""
            series = ticker.split("-", 1)[0] if "-" in ticker else ticker
        if not series:
            continue
        result.setdefault(series, {})
        result[series][norm] = {
            "yes_price": _market_yes_price(m),
            "no_price": _market_no_price(m),
            "volume": m.get("volume") or 0,
            "liquidity": m.get("liquidity"),
            "open_interest": m.get("open_interest"),
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "raw_market": m,
        }

    return result


def _parse_totals_side(market: dict) -> str | None:
    title = (market.get("title") or "").lower()
    subtitle = (
        market.get("yes_sub_title")
        or market.get("subtitle")
        or market.get("sub_title")
        or ""
    ).lower()
    if "over" in subtitle or "over" in title:
        return "OVER"
    if "under" in subtitle or "under" in title:
        return "UNDER"
    return None


def _parse_totals_line(market: dict) -> float | None:
    custom = market.get("custom_strike")
    if isinstance(custom, (int, float)):
        return float(custom)
    text = " ".join(
        [
            str(market.get("title") or ""),
            str(market.get("yes_sub_title") or ""),
            str(market.get("subtitle") or ""),
            str(market.get("sub_title") or ""),
        ]
    )
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _matchup_key(away: str | None, home: str | None) -> str | None:
    if not away or not home:
        return None
    return f"{normalize_team(away)}@{normalize_team(home)}"


def get_kalshi_totals_markets(
    target_date: date | None = None,
    tz_name: str = "US/Central",
    date_window_days: int = 0,
):
    """
    Returns a list of totals markets with parsed side/line and matchup key.
    """
    markets = []
    for ticker in KALSHI_TOTALS_SERIES_TICKERS:
        markets.extend(
            get_all_markets(
                params={"series_ticker": ticker},
                max_pages=10,
            )
        )

    now_local = datetime.now().astimezone()
    target_date = target_date or now_local.date()

    results = []
    for m in markets:
        status = (m.get("status") or "").lower()
        if status and status not in ("open", "active", "live"):
            continue

        dt_utc, dt_source = _market_game_datetime(m)
        if dt_utc:
            if dt_source == "ticker_date":
                start_date = target_date
                end_date = target_date.fromordinal(target_date.toordinal() + date_window_days)
                if not (start_date <= dt_utc.date() <= end_date):
                    continue
            else:
                dt_local = dt_utc.astimezone()
                start_date = target_date
                end_date = target_date.fromordinal(target_date.toordinal() + date_window_days)
                if not (start_date <= dt_local.date() <= end_date):
                    continue

        side = _parse_totals_side(m)
        total_line = _parse_totals_line(m)
        if not side or total_line is None:
            continue

        title = m.get("title") or ""
        away, home = _split_matchup(title)
        matchup_key = _matchup_key(away, home)

        results.append(
            {
                "side": side,
                "total": total_line,
                "price": _market_yes_price(m),
                "volume": m.get("volume") or 0,
                "ticker": m.get("ticker"),
                "event_ticker": m.get("event_ticker"),
                "matchup_key": matchup_key,
                "raw_market": m,
            }
        )

    return results


def list_ncaa_games_today(days_ahead: int = 1):
    print(f"[DEBUG] Pulling NCAA men's games for next {days_ahead} day(s)...")
    events = []
    for ticker in KALSHI_SERIES_TICKERS:
        events.extend(
            get_all_events(
                params={"series_ticker": ticker},
                max_pages=10,
            )
        )

    # Deduplicate by event ticker when we combine lists
    seen = set()
    unique_events = []
    for e in events:
        key = e.get("ticker") or id(e)
        if key in seen:
            continue
        seen.add(key)
        unique_events.append(e)

    markets = []
    for ticker in KALSHI_SERIES_TICKERS:
        markets.extend(
            get_all_markets(
                params={"series_ticker": ticker},
                max_pages=10,
            )
        )

    seen_markets = set()
    unique_markets = []
    for m in markets:
        key = m.get("ticker") or id(m)
        if key in seen_markets:
            continue
        seen_markets.add(key)
        unique_markets.append(m)

    now_local = datetime.now().astimezone()
    today_local = now_local.date()
    end_date = today_local.fromordinal(today_local.toordinal() + days_ahead)

    ncaa_today = []

    source_items = unique_markets if unique_markets else unique_events

    # Group yes/no markets into a single game using event_ticker.
    by_event = {}
    for m in source_items:
        event_key = m.get("event_ticker") or m.get("ticker") or m.get("title")
        by_event.setdefault(event_key, []).append(m)

    for markets in by_event.values():
        # Use the first market as the representative for time filtering.
        e = markets[0]
        status = (e.get("status") or "").lower()
        # Some NCAA men's events come back with missing status/time fields.
        # Treat missing status as includable so we can still surface them.
        if status and status not in ("open", "active", "live"):
            continue

        dt_utc, _ = _market_game_datetime(e)

        # Always include live games, even if date metadata is missing or off.
        if status == "live":
            dt_local = dt_utc.astimezone() if dt_utc else None
            ncaa_today.append((dt_local, markets))
            continue

        if dt_utc:
            dt_local = dt_utc.astimezone()
            if today_local <= dt_local.date() < end_date:
                ncaa_today.append((dt_local, markets))
                continue

        # If we can't evaluate the date, include it so we don't miss games.
        ncaa_today.append((None, markets))

    ncaa_today.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))

    print(f"[DEBUG] NCAA games today ({today_local}): {len(ncaa_today)}")
    if not ncaa_today:
        print("- None found")
        return []

    for dt_local, markets in ncaa_today:
        e = markets[0]
        header = _format_ncaa_row(dt_local, e)
        odds_parts = []
        for m in sorted(markets, key=lambda x: x.get("ticker") or ""):
            yes_bid = _format_price(m.get("yes_bid"))
            yes_ask = _format_price(m.get("yes_ask"))
            odds_parts.append(f"{m.get('ticker')}: {yes_bid}/{yes_ask}")
        print("-", header, "| odds:", "; ".join(odds_parts))

    return [m for _, markets in ncaa_today for m in markets]


if __name__ == "__main__":
    list_ncaa_games_today()
