import time
import re
import csv
import difflib
import math
import json
import uuid
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta

from APIs.odds_api import get_moneyline_games, get_totals_games, odds_cache_remaining
from APIs.kalshi_api import (
    get_kalshi_markets,
    get_kalshi_totals_markets,
    get_balance,
    parse_balance,
    place_order,
    get_positions,
    get_orders,
    get_trades,
    cancel_order,
    _market_game_datetime,
    get_market_by_ticker,
)
from logic.probability import american_to_prob
from logic.normalize import normalize_team
from logic.edge_calc import calculate_edge
from strategy.entry import should_enter
from alerts import alert_edge
from config.settings import (
    POLL_INTERVAL,
    KALSHI_FEE,
    MIN_EDGE,
    MIN_VOLUME,
    AGGRESSIVE_EDGE,
    AGGRESSIVE_TICK,
    KELLY_FRAC,
    KELLY_CAP,
    MAX_TRADE_PCT,
    ASSUME_LIMIT_ORDER,
    MAX_KALSHI_PRICE,
    TRADE_MODE,
    MAX_ORDERS_PER_RUN,
    POST_ONLY,
    ORDER_SLEEP_SEC,
    ORDER_UPDATE_COOLDOWN_SEC,
    MAX_UPDATES_PER_MARKET,
    ORDER_IMPROVE_MIN,
    MAX_DOLLARS_PER_MARKET,
    MAX_DOLLARS_PER_EVENT,
    MIN_EDGE_NEW,
    MIN_EDGE_ADD,
    MIN_PRICE_IMPROVEMENT_ADD,
    MAX_ADDS_PER_MARKET_PER_DAY,
    EDGE_FAV_MIN,
    EDGE_MID_MIN,
    EDGE_DOG_MIN,
    ENABLE_MONEYLINE,
    ENABLE_TOTALS,
    MONEYLINE_SIDE_FILTER,
    USE_CHEAPEST_IMPLIED_WINNER,
    ALLOW_POSITION_ADDS,
    MAX_PROB_GAP,
    CANCEL_EV_BUFFER,
    CANCEL_TIME_SEC,
    OFF_MARKET_CENTS,
    OFF_MARKET_TIME_SEC,
    CANCEL_NOT_COMPETITIVE_GAP,
    CANCEL_NOT_COMPETITIVE_TIME_SEC,
    CANCEL_NOT_COMPETITIVE_EV,
    REPLACE_EDGE_BUFFER,
    MAX_REPLACE_DRIFT,
    ALLOW_TRUE_HEDGE,
    TOTALS_LINE_TOLERANCE,
    GAME_START_CANCEL_MIN,
    OPEN_ORDERS_STATUS,
    DATE_WINDOW_DAYS,
    RESTING_CANCEL_GRACE_SEC,
    YES_NO_TIE_PREFERENCE,
    ENABLE_ARBS,
    ARB_MIN_PROFIT,
    ARB_MAX_CONTRACTS,
    ARB_MAX_ORDERS_PER_RUN,
    QUIET_LOGS,
)

PROCESS_START_TS = time.time()

if QUIET_LOGS:
    import builtins as _builtins

    _SKIP_COUNTS = Counter()
    _EDGE_COUNTS = Counter()

    def _record_skip(msg: str) -> None:
        lower = msg.lower()
        if "resting exposure" in lower or "duplicate" in lower or "already bet" in lower:
            _SKIP_COUNTS["dup"] += 1
            return
        if "new edge" in lower:
            _SKIP_COUNTS["edge_new_small"] += 1
            return
        if "add edge" in lower:
            _SKIP_COUNTS["edge_add_small"] += 1
            return
        if "edge<" in lower:
            _SKIP_COUNTS["edge_small"] += 1
            return
        if "vol<" in lower:
            _SKIP_COUNTS["volume"] += 1
            return
        if "game started" in lower:
            _SKIP_COUNTS["game_started"] += 1
            return
        if "no bid" in lower:
            _SKIP_COUNTS["no_bid"] += 1
            return
        if "market cap" in lower or "event cap" in lower:
            _SKIP_COUNTS["cap"] += 1
            return
        if "add not better" in lower:
            _SKIP_COUNTS["add_not_better"] += 1
            return
        if "update improve" in lower or "update cooldown" in lower or "max updates" in lower:
            _SKIP_COUNTS["update"] += 1
            return
        if "max adds" in lower or "add disabled" in lower:
            _SKIP_COUNTS["add_limit"] += 1
            return
        if "missing p_true" in lower or "missing implied winner" in lower:
            _SKIP_COUNTS["missing_prob"] += 1
            return
        if "missing market" in lower or "zero contracts" in lower:
            _SKIP_COUNTS["bad_order"] += 1
            return
        _SKIP_COUNTS["other"] += 1

    def _skip_summary_line() -> str:
        counts = _skip_summary_counts()
        total = sum(counts.values())
        if total == 0:
            return "[INFO] Summary: no skips"
        parts = [f"{k}={v}" for k, v in counts.items()]
        return "[INFO] Summary: " + " | ".join(parts)

    def _skip_summary_counts() -> dict:
        return {
            "dup": _SKIP_COUNTS["dup"],
            "new_small": _SKIP_COUNTS["edge_new_small"],
            "add_small": _SKIP_COUNTS["edge_add_small"],
            "edge_small": _SKIP_COUNTS["edge_small"],
            "add_not_better": _SKIP_COUNTS["add_not_better"],
            "game_started": _SKIP_COUNTS["game_started"],
            "no_bid": _SKIP_COUNTS["no_bid"],
            "caps": _SKIP_COUNTS["cap"],
            "updates": _SKIP_COUNTS["update"],
            "volume": _SKIP_COUNTS["volume"],
            "missing": _SKIP_COUNTS["missing_prob"] + _SKIP_COUNTS["bad_order"],
            "other": _SKIP_COUNTS["other"],
        }

    def _edge_summary_line() -> str:
        total = _EDGE_COUNTS.get("total", 0)
        if total == 0:
            return "[INFO] Edges: 0"
        return (
            "[INFO] Edges: "
            f"total={total} | "
            f">=5%={_EDGE_COUNTS.get('ge_5', 0)} | "
            f">=3%={_EDGE_COUNTS.get('ge_3', 0)} | "
            f">=2%={_EDGE_COUNTS.get('ge_2', 0)} | "
            f">=1%={_EDGE_COUNTS.get('ge_1', 0)}"
        )

    _ALLOW_PREFIXES = (
        "[INFO] Run",
        "[INFO] Placed order",
        "[INFO] DRY RUN order",
        "[INFO] Placed ARB",
        "[INFO] DRY RUN ARB",
        "[INFO] In-market arbs",
        "[ARB]",
        "[INFO] Odds API refresh in",
        "[INFO] Edges",
        "[INFO] Matched rows",
        "[INFO] Unmatched",
        "[INFO] Summary",
        "[INFO] Canceled",
        "[INFO] Cancel not found",
        "[WARN]",
        "[ERROR]",
    )

    def _filtered_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if msg.startswith("[SKIP]"):
            _record_skip(msg)
        if msg.startswith(_ALLOW_PREFIXES):
            _builtins.print(*args, **kwargs)

    print = _filtered_print

def kalshi_fee_per_contract(price: float) -> float:
    return math.ceil(0.07 * price * (1 - price) * 100) / 100


def expected_profit_per_contract(book_prob: float, price: float, is_limit_order: bool) -> float:
    fee = 0 if is_limit_order else kalshi_fee_per_contract(price)
    return book_prob * (1 - price - fee) - (1 - book_prob) * price


def ev_per_dollar(book_prob: float, kalshi_price: float) -> float:
    # EV per $ risked: (q*(1-p) - (1-q)*p)/p where q=book_prob, p=kalshi_price
    q = book_prob
    p = kalshi_price
    if p <= 0:
        return 0.0
    return (q * (1 - p) - (1 - q) * p) / p
    


def _devig_probs(odds_a: int, odds_b: int) -> tuple[float, float]:
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    vig = pa + pb
    if vig <= 0:
        return pa, pb
    return pa / vig, pb / vig


def _min_edge_for_prob(p_true: float) -> float:
    if p_true > 0.70:
        return EDGE_FAV_MIN
    if p_true < 0.40:
        return EDGE_DOG_MIN
    return EDGE_MID_MIN


def _moneyline_side_allowed(side: str) -> bool:
    if USE_CHEAPEST_IMPLIED_WINNER:
        return True
    filt = (MONEYLINE_SIDE_FILTER or "ALL").upper()
    if filt == "ALL":
        return True
    return side.upper() == filt


def _format_odds(odds: float | int) -> str:
    try:
        val = float(odds)
    except Exception:
        return str(odds)
    if val.is_integer():
        return f"{val:+.0f}"
    return f"{val:+.1f}"


def _prob_to_american(prob: float | None) -> int | None:
    if prob is None:
        return None
    try:
        prob = float(prob)
    except Exception:
        return None
    if prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def _extract_order_id(resp: dict | None) -> str | None:
    if not isinstance(resp, dict):
        return None
    if resp.get("order_id"):
        return str(resp["order_id"])
    if resp.get("id"):
        return str(resp["id"])
    order = resp.get("order")
    if isinstance(order, dict) and order.get("order_id"):
        return str(order["order_id"])
    return None

def _series_for_sport(sport_key: str | None) -> str | None:
    if not sport_key:
        return None
    mapping = {
        "basketball_ncaab": "KXNCAAMBGAME",
        "basketball_wncaab": "KXNCAAWBGAME",
    }
    return mapping.get(sport_key)


def _series_from_ticker(ticker: str | None) -> str:
    if not ticker:
        return ""
    return ticker.split("-", 1)[0] if "-" in ticker else ticker


def _matchup_key(away: str, home: str) -> str:
    return f"{normalize_team(away)}@{normalize_team(home)}"


def _find_totals_market(
    totals_list: list[dict],
    matchup_key: str,
    side: str,
    total_line: float,
    tolerance: float = TOTALS_LINE_TOLERANCE,
) -> dict | None:
    candidates = [
        m
        for m in totals_list
        if m.get("matchup_key") == matchup_key and m.get("side") == side
    ]
    if not candidates:
        return None
    best = None
    best_diff = 10**9
    for m in candidates:
        line = m.get("total")
        if line is None:
            continue
        diff = abs(float(line) - float(total_line))
        if diff < best_diff:
            best = m
            best_diff = diff
    if best and best_diff <= tolerance:
        return best
    return None


def _parse_commence_time(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _odds_start_times(games: list[dict]) -> dict:
    out = {}
    for g in games:
        key = _matchup_key(g["away"], g["home"])
        dt = _parse_commence_time(g.get("commence_time"))
        if key and dt:
            out[key] = dt
            # Also store reversed order to handle "vs" titles where home/away is unclear.
            rev_key = _matchup_key(g["home"], g["away"])
            out[rev_key] = dt
    return out


def _prob_band(p_true: float) -> str:
    if p_true > 0.70:
        return "fav"
    if p_true < 0.40:
        return "dog"
    return "mid"


def stake_size(
    bankroll,
    book_prob,
    kalshi_price,
    contracts_available,
    is_limit_order=True,
    kelly_frac=0.125,
    max_pct=0.10,
):
    fee = 0 if (is_limit_order and POST_ONLY) else kalshi_fee_per_contract(kalshi_price)
    effective_price = kalshi_price + fee
    if effective_price <= 0:
        return 0

    b = (1 / effective_price) - 1
    p = book_prob
    q = 1 - p

    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0
    kelly = min(kelly, KELLY_CAP)

    raw_stake = bankroll * kelly * kelly_frac
    capped_stake = min(raw_stake, bankroll * max_pct)
    contracts = int(capped_stake / effective_price)

    return min(contracts, contracts_available)


LEDGER_PATH = Path("data/bets_ledger.json")
RESTING_PATH = Path("data/resting_orders.json")
RESTING_SNAPSHOT_PATH = Path("data/resting_orders_snapshot.json")
POSITIONS_SNAPSHOT_PATH = Path("data/positions_snapshot.json")
FILLED_TRADES_PATH = Path("data/filled_trades.csv")
ORDERS_LOG_PATH = Path("data/orders_log.csv")


def load_ledger():
    if not LEDGER_PATH.exists():
        return {}
    try:
        return json.loads(LEDGER_PATH.read_text())
    except Exception:
        return {}


def save_ledger(data):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    def _stringify_keys(obj):
        if isinstance(obj, dict):
            return {str(k): _stringify_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_stringify_keys(v) for v in obj]
        return obj

    safe = _stringify_keys(data)
    LEDGER_PATH.write_text(json.dumps(safe, indent=2, sort_keys=True))




def save_resting_orders_snapshot(data):
    RESTING_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESTING_SNAPSHOT_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def save_positions_snapshot(data):
    POSITIONS_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_SNAPSHOT_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def _append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _load_csv_ids(path: Path, id_field: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return {row.get(id_field, "") for row in reader if row.get(id_field)}
    except Exception:
        return set()


def _load_orders_log_by_id() -> dict[str, dict]:
    if not ORDERS_LOG_PATH.exists():
        return {}
    try:
        with ORDERS_LOG_PATH.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return {row.get("order_id", ""): row for row in reader if row.get("order_id")}
    except Exception:
        return {}


def _log_order_entry(
    order_id: str | None,
    market_ticker: str,
    side: str,
    count: float,
    price: float,
    edge: float | None,
    p_true: float | None,
    book_odds: float | None,
    market_type: str | None,
    total_line: float | None,
    team: str | None,
    event_ticker: str | None,
) -> None:
    if not order_id:
        return
    row = {
        "timestamp": datetime.now().isoformat(),
        "order_id": order_id,
        "market_ticker": market_ticker,
        "side": side,
        "count": count,
        "price": price,
        "edge": round(edge, 6) if edge is not None else "",
        "p_true": round(p_true, 6) if p_true is not None else "",
        "book_odds": book_odds if book_odds is not None else "",
        "market_type": market_type or "",
        "total_line": total_line if total_line is not None else "",
        "team": team or "",
        "event_ticker": event_ticker or "",
    }
    _append_csv_rows(
        ORDERS_LOG_PATH,
        [
            "timestamp",
            "order_id",
            "market_ticker",
            "side",
            "count",
            "price",
            "edge",
            "p_true",
            "book_odds",
            "market_type",
            "total_line",
            "team",
            "event_ticker",
        ],
        [row],
    )


def _log_filled_trades(trades: list[dict]) -> None:
    if not trades:
        return
    existing_ids = _load_csv_ids(FILLED_TRADES_PATH, "trade_id")
    order_meta = _load_orders_log_by_id()
    rows = []
    for t in trades:
        trade_id = t.get("trade_id") or t.get("fill_id")
        if not trade_id or trade_id in existing_ids:
            continue
        order_id = t.get("order_id") or ""
        price = _norm_price(t.get("price"))
        if price is None:
            price = _norm_price(t.get("yes_price_dollars") or t.get("no_price_dollars"))
        if price is None:
            price = _norm_price(t.get("yes_price_fixed") or t.get("no_price_fixed"))
        fee_cost = _norm_dollars(t.get("fee_cost")) or 0.0
        count = t.get("count") or t.get("count_fp") or 0
        try:
            count = float(count)
        except Exception:
            count = 0.0
        meta = order_meta.get(str(order_id)) or {}
        rows.append(
            {
                "fill_time": t.get("created_time") or t.get("ts") or "",
                "trade_id": trade_id,
                "order_id": order_id,
                "market_ticker": t.get("market_ticker") or t.get("ticker") or "",
                "side": (t.get("side") or "").upper(),
                "action": t.get("action") or "",
                "count": count,
                "price": price if price is not None else "",
                "fee_cost": fee_cost,
                "is_taker": t.get("is_taker"),
                "edge_at_entry": meta.get("edge", ""),
                "p_true_at_entry": meta.get("p_true", ""),
                "book_odds": meta.get("book_odds", ""),
                "market_type": meta.get("market_type", ""),
                "total_line": meta.get("total_line", ""),
                "team": meta.get("team", ""),
                "event_ticker": meta.get("event_ticker", ""),
            }
        )
    _append_csv_rows(
        FILLED_TRADES_PATH,
        [
            "fill_time",
            "trade_id",
            "order_id",
            "market_ticker",
            "side",
            "action",
            "count",
            "price",
            "fee_cost",
            "is_taker",
            "edge_at_entry",
            "p_true_at_entry",
            "book_odds",
            "market_type",
            "total_line",
            "team",
            "event_ticker",
        ],
        rows,
    )


def _sorted_snapshot_list(items):
    if not isinstance(items, list):
        return items
    def _key(x):
        if not isinstance(x, dict):
            return ""
        return (
            x.get("market_ticker")
            or x.get("ticker")
            or x.get("event_ticker")
            or ""
        )
    return sorted(items, key=_key)


def load_resting_state():
    if not RESTING_PATH.exists():
        return {}
    try:
        return json.loads(RESTING_PATH.read_text())
    except Exception:
        return {}


def save_resting_state(data):
    RESTING_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESTING_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def _price_to_dollars(price: float) -> str:
    return f"{price:.4f}"


def _order_key(market_ticker: str, side: str, target_date: str) -> str:
    return f"{market_ticker}:{side}:{target_date}"


def _matchup_teams(market: dict) -> tuple[str | None, str | None]:
    title = (market or {}).get("title") or ""
    clean = re.sub(r"\bWinner\??\b", "", title, flags=re.IGNORECASE).strip()
    # Strip totals suffix like ": Total Points"
    if ":" in clean:
        clean = clean.split(":", 1)[0].strip()
    if " at " in clean:
        away, home = clean.split(" at ", 1)
        return _clean_team_name(away), _clean_team_name(home)
    if " vs " in clean:
        t1, t2 = clean.split(" vs ", 1)
        return _clean_team_name(t1), _clean_team_name(t2)
    return None, None


def _clean_team_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = name.strip().strip(":-?.,")
    return normalize_team(cleaned)


def _implied_winner(team: str, side: str, market: dict) -> str | None:
    title = (market or {}).get("title", "").lower()
    if "total" in title or "over" in title or "under" in title:
        return None
    team_norm = normalize_team(team)
    if side == "YES":
        return team_norm
    away, home = _matchup_teams(market)
    away_norm = normalize_team(away) if away else None
    home_norm = normalize_team(home) if home else None
    if team_norm == away_norm:
        return home_norm
    if team_norm == home_norm:
        return away_norm
    return None


def _market_by_ticker(kalshi_data: dict) -> dict:
    by_ticker = {}
    for v in kalshi_data.values():
        if isinstance(v, dict) and v.get("ticker"):
            t = v.get("ticker")
            by_ticker[t] = v.get("raw_market")
            continue
        if isinstance(v, dict):
            for inner in v.values():
                t = inner.get("ticker")
                if t:
                    by_ticker[t] = inner.get("raw_market")
    return by_ticker


def _event_from_market_ticker(ticker: str | None) -> str | None:
    if not ticker:
        return None
    if "-" not in ticker:
        return ticker
    return ticker.rsplit("-", 1)[0]


def _ticker_to_team(kalshi_data: dict) -> dict:
    out = {}
    for team, v in kalshi_data.items():
        if isinstance(v, dict) and v.get("ticker"):
            t = v.get("ticker")
            out[t] = team
            continue
        if isinstance(v, dict):
            for inner_team, inner in v.items():
                t = inner.get("ticker")
                if t:
                    out[t] = inner_team
    return out


def _event_team_yes_prices(kalshi_data: dict) -> dict:
    out: dict[str, dict[str, float]] = {}
    for series_map in kalshi_data.values():
        if not isinstance(series_map, dict):
            continue
        for team, info in series_map.items():
            if not isinstance(info, dict):
                continue
            event = info.get("event_ticker")
            raw_market = info.get("raw_market") or {}
            price = raw_market.get("yes_bid")
            if price is None:
                price = info.get("yes_price")
            if price is None:
                continue
            try:
                price = float(price)
            except Exception:
                continue
            if price > 1:
                price = price / 100
            if not event:
                continue
            team_norm = normalize_team(team)
            out.setdefault(event, {})[team_norm] = price
    return out


def _choose_implied_candidate(
    current: tuple[float, str] | None,
    candidate: tuple[float, str],
    threshold: float,
) -> tuple[float, str]:
    if current is None:
        return candidate
    cur_price, cur_side = current
    cand_price, cand_side = candidate
    if abs(cand_price - cur_price) < threshold:
        if cand_side == "YES" and cur_side != "YES":
            return candidate
        if cur_side == "YES" and cand_side != "YES":
            return current
    if cand_price < cur_price:
        return candidate
    return current


def _is_preferred_implied(
    price: float,
    side: str,
    best_price: float | None,
    best_side: str | None,
    threshold: float,
) -> bool:
    if best_price is None:
        return True
    if abs(price - best_price) < threshold:
        if side == "YES" and best_side != "YES":
            return True
        if best_side == "YES" and side != "YES":
            return False
    return price <= best_price


def _best_bid_for_row(row: dict, market: dict) -> float | None:
    side = row.get("side")
    if not side:
        return None
    bid = _post_only_price(market, side)
    return bid


def _extract_side(item: dict) -> str | None:
    side = (item.get("side") or "").lower()
    if side in ("yes", "no"):
        return side.upper()
    return None


def _position_side(item: dict) -> str | None:
    side = _extract_side(item)
    if side:
        return side
    pos = item.get("position") or item.get("count") or 0
    try:
        pos = float(pos)
    except Exception:
        return None
    if pos < 0:
        return "NO"
    if pos > 0:
        return "YES"
    return None


def _positions_exposure(
    positions_json: dict, market_map: dict, ticker_to_team: dict
) -> set:
    data = (
        positions_json.get("market_positions")
        or positions_json.get("positions")
        or positions_json.get("data")
        or []
    )
    exposures = set()
    for p in data:
        count = p.get("count") or p.get("position") or 0
        try:
            count = abs(float(count))
        except Exception:
            count = 0
        if count <= 0:
            continue
        side = _position_side(p)
        ticker = p.get("market_ticker") or p.get("ticker")
        if not side or not ticker:
            continue
        market = market_map.get(ticker)
        # infer team if present
        team = p.get("team") or p.get("team_name") or ticker_to_team.get(ticker) or ""
        implied = _implied_winner(normalize_team(team), side, market) if team else None
        key = (ticker, implied, side)
        exposures.add(key)
    return exposures


def _positions_tickers(positions_json: dict) -> set:
    data = (
        positions_json.get("market_positions")
        or positions_json.get("positions")
        or positions_json.get("data")
        or []
    )
    tickers = set()
    for p in data:
        ticker = p.get("market_ticker") or p.get("ticker")
        if ticker:
            tickers.add(ticker)
    return tickers


def _positions_by_market(positions_json: dict) -> dict:
    data = (
        positions_json.get("market_positions")
        or positions_json.get("positions")
        or positions_json.get("data")
        or []
    )
    out = {}
    for p in data:
        market_info = p.get("market") or {}
        ticker = (
            p.get("market_ticker")
            or p.get("ticker")
            or market_info.get("market_ticker")
            or market_info.get("ticker")
        )
        if not ticker:
            continue
        count = (
            p.get("count")
            or p.get("position")
            or p.get("quantity")
            or p.get("size")
            or 0
        )
        if isinstance(count, dict):
            count = count.get("count") or count.get("position") or 0
        try:
            count = abs(float(count))
        except Exception:
            count = 0
        if count <= 0:
            continue
        avg_price = (
            p.get("avg_price")
            or p.get("average_price")
            or p.get("average_entry_price")
            or p.get("avg_cost")
            or p.get("avg_price_dollars")
            or p.get("avg_cost_dollars")
            or p.get("price")
            or p.get("cost")
        )
        avg_price = _norm_price(avg_price)
        fees_paid = (
            p.get("fees_paid_dollars")
            or p.get("fees_paid")
            or 0
        )
        fees_paid = _norm_dollars(fees_paid) or 0.0
        if avg_price is None:
            exposure_dollars = (
                p.get("market_exposure_dollars")
                or p.get("total_traded_dollars")
                or p.get("market_exposure")
                or p.get("total_traded")
            )
            exposure_dollars = _norm_dollars(exposure_dollars)
            if exposure_dollars is not None and count > 0:
                avg_price = exposure_dollars / count
        if avg_price is None:
            continue
        dollars = avg_price * count
        entry = out.setdefault(ticker, {"count": 0.0, "dollars": 0.0, "avg_price": 0.0})
        entry["count"] += count
        entry["dollars"] += dollars
        entry["avg_price"] = (
            entry["dollars"] / entry["count"] if entry["count"] > 0 else avg_price
        )
    return out


def _positions_by_event(positions_json: dict, market_map: dict) -> dict:
    data = (
        positions_json.get("market_positions")
        or positions_json.get("positions")
        or positions_json.get("data")
        or []
    )
    out = {}
    for p in data:
        market_info = p.get("market") or {}
        ticker = (
            p.get("market_ticker")
            or p.get("ticker")
            or market_info.get("market_ticker")
            or market_info.get("ticker")
        )
        if not ticker:
            continue
        market = market_map.get(ticker) or {}
        event = market.get("event_ticker") or ticker
        count = (
            p.get("count")
            or p.get("position")
            or p.get("quantity")
            or p.get("size")
            or 0
        )
        if isinstance(count, dict):
            count = count.get("count") or count.get("position") or 0
        try:
            count = abs(float(count))
        except Exception:
            count = 0
        avg_price = (
            p.get("avg_price")
            or p.get("average_price")
            or p.get("average_entry_price")
            or p.get("price")
            or p.get("avg_cost")
            or 0
        )
        avg_price = _norm_price(avg_price)
        if avg_price is None:
            exposure_dollars = (
                p.get("market_exposure_dollars")
                or p.get("total_traded_dollars")
                or p.get("market_exposure")
                or p.get("total_traded")
            )
            exposure_dollars = _norm_dollars(exposure_dollars)
            if exposure_dollars is not None and count > 0:
                avg_price = exposure_dollars / count
        avg_price = avg_price or 0
        out[event] = out.get(event, 0) + (count * avg_price)
    return out


def _orders_exposure(
    orders_json: dict, market_map: dict, ticker_to_team: dict
) -> set:
    data = orders_json.get("orders") or orders_json.get("data") or []
    exposures = set()
    for o in data:
        status = (o.get("status") or "").lower()
        if status and status not in ("open", "resting", "active", "placed", "live"):
            continue
        side = _extract_side(o)
        ticker = o.get("market_ticker") or o.get("ticker")
        if not side or not ticker:
            continue
        market = market_map.get(ticker)
        team = o.get("team") or o.get("team_name") or ticker_to_team.get(ticker) or ""
        implied = _implied_winner(normalize_team(team), side, market) if team else None
        key = (ticker, implied, side)
        exposures.add(key)
    return exposures


def _event_implied_map(exposures: set, market_map: dict) -> dict:
    out = {}
    for ticker, implied, _side in exposures:
        if not implied:
            continue
        market = market_map.get(ticker) or {}
        event = market.get("event_ticker") or ticker
        out.setdefault(event, set()).add(implied)
    return out


def _order_price(item: dict) -> float | None:
    if "yes_price_fixed" in item:
        try:
            return float(item["yes_price_fixed"])
        except Exception:
            pass
    if "no_price_fixed" in item:
        try:
            return float(item["no_price_fixed"])
        except Exception:
            pass
    if "price" in item:
        try:
            return float(item["price"])
        except Exception:
            pass
    if "yes_price" in item:
        return _norm_price(item.get("yes_price"))
    if "no_price" in item:
        return _norm_price(item.get("no_price"))
    return None


def _open_orders_map(orders_json: dict) -> dict:
    data = orders_json.get("orders") or orders_json.get("data") or []
    out = {}
    for o in data:
        status = (o.get("status") or "").lower()
        if status and status not in ("open", "resting", "active", "placed", "live"):
            continue
        remaining = o.get("remaining_count") or o.get("open_count") or o.get("count") or 0
        try:
            remaining = float(remaining)
        except Exception:
            remaining = 0
        if remaining <= 0:
            continue
        side = _extract_side(o)
        ticker = o.get("market_ticker") or o.get("ticker")
        order_id = o.get("order_id") or o.get("id")
        price = _order_price(o)
        if not (side and ticker and order_id and price is not None):
            continue
        out[(ticker, side)] = {
            "order_id": order_id,
            "price": price,
        }
    return out


def _holdings_from_positions(positions_json: dict, market_map: dict, ticker_to_team: dict):
    data = (
        positions_json.get("market_positions")
        or positions_json.get("positions")
        or positions_json.get("data")
        or []
    )
    holdings = []
    for p in data:
        count = p.get("count") or p.get("position") or 0
        try:
            count = float(count)
        except Exception:
            count = 0
        if count == 0:
            continue
        side = _position_side(p)
        avg_price = (
            p.get("avg_price")
            or p.get("average_price")
            or p.get("average_entry_price")
            or p.get("avg_cost")
            or p.get("avg_price_dollars")
            or p.get("avg_cost_dollars")
            or p.get("price")
            or p.get("cost")
        )
        avg_price = _norm_price(avg_price)
        fees_paid = (
            p.get("fees_paid_dollars")
            or p.get("fees_paid")
            or 0
        )
        fees_paid = _norm_dollars(fees_paid) or 0.0
        if avg_price is None:
            exposure_dollars = (
                p.get("market_exposure_dollars")
                or p.get("total_traded_dollars")
                or p.get("market_exposure")
                or p.get("total_traded")
            )
            exposure_dollars = _norm_dollars(exposure_dollars)
            if exposure_dollars is not None and count:
                avg_price = exposure_dollars / abs(count)
        ticker = p.get("market_ticker") or p.get("ticker")
        if not side or not ticker:
            continue
        market = market_map.get(ticker)
        team = ticker_to_team.get(ticker)
        event_ticker = (market or {}).get("event_ticker") or _event_from_market_ticker(ticker)
        holdings.append(
            {
                "source": "position",
                "ticker": ticker,
                "side": side,
                "team": team,
                "event_ticker": event_ticker,
                "count": abs(count),
                "price": avg_price,
                "fees": fees_paid,
            }
        )
    return holdings


def _holdings_from_orders(orders_json: dict, market_map: dict, ticker_to_team: dict):
    data = orders_json.get("orders") or orders_json.get("data") or []
    holdings = []
    for o in data:
        status = (o.get("status") or "").lower()
        if status and status not in ("open", "resting", "active", "placed", "live"):
            continue
        remaining = o.get("remaining_count") or o.get("open_count") or o.get("count") or 0
        try:
            remaining = float(remaining)
        except Exception:
            remaining = 0
        if remaining <= 0:
            continue
        side = _extract_side(o)
        ticker = o.get("market_ticker") or o.get("ticker")
        if not side or not ticker:
            continue
        price = _order_price(o)
        market = market_map.get(ticker)
        team = ticker_to_team.get(ticker)
        event_ticker = (market or {}).get("event_ticker") or _event_from_market_ticker(ticker)
        holdings.append(
            {
                "source": "order",
                "ticker": ticker,
                "side": side,
                "team": team,
                "event_ticker": event_ticker,
                "count": remaining,
                "price": price,
            }
        )
    return holdings


def _report_arbs(holdings: list, market_map: dict | None = None):
    # In-market arb: YES + NO in same market_ticker
    by_market = {}
    for h in holdings:
        if h.get("source") != "position":
            continue
        t = h.get("ticker")
        s = h.get("side")
        if not t or not s:
            continue
        by_market.setdefault(t, set()).add(s)

    in_market = [t for t, sides in by_market.items() if "YES" in sides and "NO" in sides]
    arbs = []
    hedges = []
    for t in in_market:
        yes_ask = None
        no_ask = None
        profit = None
        if market_map and t in market_map:
            m = market_map.get(t) or {}
            yes_ask = _norm_price(m.get("yes_ask"))
            no_ask = _norm_price(m.get("no_ask"))
            if yes_ask is not None and no_ask is not None:
                fee_yes = kalshi_fee_per_contract(yes_ask)
                fee_no = kalshi_fee_per_contract(no_ask)
                profit = 1 - (yes_ask + no_ask + fee_yes + fee_no)
        if profit is not None and profit >= 0:
            arbs.append((t, yes_ask, no_ask, profit))
        else:
            hedges.append((t, yes_ask, no_ask, profit))

    # Game-level hedge: YES/YES or NO/NO across both teams in same event
    by_event = {}
    event_positions = {}
    event_teams = {}
    if market_map:
        for t, m in market_map.items():
            ev_key = (m or {}).get("event_ticker") or t
            away, home = _matchup_teams(m or {})
            if away and home:
                event_teams.setdefault(ev_key, set()).update(
                    {normalize_team(away), normalize_team(home)}
                )
    for h in holdings:
        if h.get("source") != "position":
            continue
        ev = h.get("event_ticker") or h.get("ticker")
        team = h.get("team")
        side = h.get("side")
        price = h.get("price")
        count = h.get("count")
        if not ev or not team or not side:
            continue
        by_event.setdefault(ev, {"YES": set(), "NO": set()})
        by_event[ev][side].add(team)
        event_teams.setdefault(ev, set()).add(normalize_team(team))
        market = market_map.get(h.get("ticker")) if market_map else None
        implied = _implied_winner(team, side, market) if market else None
        if implied is None and side == "YES":
            implied = normalize_team(team)
        if implied is None and side == "NO":
            teams_set = event_teams.get(ev) or set()
            team_norm = normalize_team(team)
            others = [t for t in teams_set if t and t != team_norm]
            if len(others) == 1:
                implied = others[0]
        if implied and price is not None and count:
            event_positions.setdefault(ev, []).append(
                {
                    "implied": implied,
                    "price": float(price),
                    "count": float(count),
                    "fees": float(h.get("fees") or 0),
                }
            )
            event_teams.setdefault(ev, set()).add(normalize_team(team))

    game_level = []
    for ev, sides in by_event.items():
        if len(sides["YES"]) >= 2:
            game_level.append((ev, "YES", sorted(sides["YES"])))
        if len(sides["NO"]) >= 2:
            game_level.append((ev, "NO", sorted(sides["NO"])))

    print(f"[INFO] In-market arbs: {len(arbs)}")
    for t, yes_ask, no_ask, profit in arbs[:50]:
        print(
            f"[ARB] market {t}: YES@{yes_ask:.2f} + NO@{no_ask:.2f} "
            f"| profit {profit:.4f} (fees included)"
        )
    if hedges:
        print(f"[INFO] In-market both sides (not arbs): {len(hedges)}")
        for t, yes_ask, no_ask, profit in hedges[:50]:
            if yes_ask is not None and no_ask is not None and profit is not None:
                print(
                    f"[HEDGE] market {t}: YES@{yes_ask:.2f} + NO@{no_ask:.2f} "
                    f"| profit {profit:.4f} (fees included)"
                )
            else:
                print(f"[HEDGE] market {t}: YES + NO")

    print(f"[INFO] Game-level hedges: {len(game_level)}")
    for ev, side, teams in game_level[:50]:
        pnl_info = ""
        teams_set = event_teams.get(ev) or set(normalize_team(t) for t in teams)
        teams_list = sorted(t for t in teams_set if t)
        if len(teams_list) >= 2 and ev in event_positions:
            t1, t2 = teams_list[0], teams_list[1]
            fees_total = sum(p.get("fees", 0.0) for p in event_positions.get(ev, []))
            def _pnl_for(winner: str) -> float:
                pnl = 0.0
                for pos in event_positions.get(ev, []):
                    if pos["implied"] == winner:
                        pnl += pos["count"] * (1 - pos["price"])
                    else:
                        pnl -= pos["count"] * pos["price"]
                return pnl - fees_total
            pnl1 = _pnl_for(t1)
            pnl2 = _pnl_for(t2)
            pnl_info = f" | P/L if {t1} wins: {pnl1:.2f}, if {t2} wins: {pnl2:.2f}"
        if pnl_info and pnl1 >= 0 and pnl2 >= 0:
            print(f"[ARB] event {ev} {side}: {', '.join(teams)}{pnl_info}")
        else:
            print(f"[HEDGE] {ev} {side}: {', '.join(teams)}{pnl_info}")


def _norm_price(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
        value = float(value)
    except Exception:
        return None
    return value / 100 if value > 1 else value


def _norm_dollars(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
        return float(value)
    except Exception:
        return None


def _post_only_price(market: dict, side: str) -> float | None:
    if not market:
        return None
    if side == "YES":
        return _norm_price(market.get("yes_bid"))
    return _norm_price(market.get("no_bid"))


def _mid_price(market: dict, side: str) -> float | None:
    if not market:
        return None
    if side == "YES":
        bid = _norm_price(market.get("yes_bid"))
        ask = _norm_price(market.get("yes_ask"))
    else:
        bid = _norm_price(market.get("no_bid"))
        ask = _norm_price(market.get("no_ask"))
    if bid is None and ask is None:
        return None
    if bid is None:
        return ask
    if ask is None:
        return bid
    return (bid + ask) / 2


def _best_limit_price(market: dict, side: str) -> float | None:
    if not market:
        return None
    if side == "YES":
        bid = _norm_price(market.get("yes_bid"))
        ask = _norm_price(market.get("yes_ask"))
    else:
        bid = _norm_price(market.get("no_bid"))
        ask = _norm_price(market.get("no_ask"))

    if bid is None:
        return None

    if ask is not None:
        spread = ask - bid
        if spread <= 0.01:
            limit_price = bid
        else:
            limit_price = bid + 0.01
            # Ensure we don't cross due to rounding.
            if limit_price >= ask:
                limit_price = bid
    else:
        limit_price = bid + 0.01

    # HARD safety rails
    limit_price = max(0.01, min(0.99, limit_price))
    limit_price = round(limit_price, 2)
    return limit_price


def _edge_based_limit_price(
    market: dict,
    side: str,
    edge: float | None,
    min_edge: float,
    aggressive_edge: float,
    tick: float,
) -> float | None:
    if not market:
        return None
    if edge is None:
        return None
    if edge < min_edge:
        return None
    if side == "YES":
        bid = _norm_price(market.get("yes_bid"))
        ask = _norm_price(market.get("yes_ask"))
    else:
        bid = _norm_price(market.get("no_bid"))
        ask = _norm_price(market.get("no_ask"))
    if bid is None:
        return None
    if edge >= aggressive_edge:
        limit_price = bid + tick
        if ask is not None and limit_price >= ask:
            limit_price = bid
    else:
        limit_price = bid
    limit_price = max(0.01, min(0.99, limit_price))
    limit_price = round(limit_price, 2)
    return limit_price


def _arb_candidates(market_map: dict) -> list[dict]:
    out = []
    for ticker, market in (market_map or {}).items():
        if not isinstance(market, dict):
            continue
        title = (market.get("title") or "").lower()
        if "total" in title or "over" in title or "under" in title:
            continue
        yes_ask = _norm_price(market.get("yes_ask"))
        no_ask = _norm_price(market.get("no_ask"))
        if yes_ask is None or no_ask is None:
            continue
        fee_yes = kalshi_fee_per_contract(yes_ask)
        fee_no = kalshi_fee_per_contract(no_ask)
        total = yes_ask + no_ask + fee_yes + fee_no
        profit = 1 - total
        if profit >= ARB_MIN_PROFIT:
            out.append(
                {
                    "ticker": ticker,
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                    "fee_yes": fee_yes,
                    "fee_no": fee_no,
                    "total": total,
                    "profit": profit,
                }
            )
    return out


def _market_started(market: dict, odds_start_by_matchup: dict | None) -> bool:
    if market:
        status = (market.get("status") or "").lower()
        if status in ("live", "in_play", "inplay", "closed", "settled"):
            return True
    if market and odds_start_by_matchup:
        away, home = _matchup_teams(market)
        if away and home:
            key = _matchup_key(away, home)
            dt_utc = odds_start_by_matchup.get(key)
            if dt_utc:
                now_utc = datetime.utcnow().replace(tzinfo=dt_utc.tzinfo)
                cutoff = dt_utc - timedelta(minutes=GAME_START_CANCEL_MIN)
                return now_utc >= cutoff

    if not market:
        return False
    dt_utc, source = _market_game_datetime(market)
    if not dt_utc:
        return False
    now_utc = datetime.utcnow().replace(tzinfo=dt_utc.tzinfo)
    if source == "ticker_date":
        return dt_utc.date() < now_utc.date()
    cutoff = dt_utc - timedelta(minutes=GAME_START_CANCEL_MIN)
    return now_utc >= cutoff


def run():
    while True:
        if QUIET_LOGS:
            _SKIP_COUNTS.clear()
            _EDGE_COUNTS.clear()
            now_label = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[INFO] Run {now_label}")
            print(f"[INFO] Bot file: {Path(__file__).resolve()} | build=series-counts-2026-02-07")
        print("[INFO] Fetching odds + Kalshi data...")
        target_date = datetime.now().astimezone().date()
        target_date_str = target_date.isoformat()
        games = []
        totals_games = []
        if ENABLE_MONEYLINE:
            games = get_moneyline_games(
                target_date=target_date,
                tz_name="US/Central",
                date_window_days=DATE_WINDOW_DAYS,
            )
        if ENABLE_TOTALS:
            totals_games = get_totals_games(
                target_date=target_date,
                tz_name="US/Central",
                date_window_days=DATE_WINDOW_DAYS,
            )
        if games:
            sample_games = ", ".join(
                f"{g['away']} at {g['home']}" for g in games[:5]
            )
            print(f"[INFO] Odds moneyline sample: {sample_games}")
        if totals_games:
            sample_totals = ", ".join(
                f"{g['away']} at {g['home']} {g['total']}" for g in totals_games[:5]
            )
            print(f"[INFO] Odds totals sample: {sample_totals}")
        odds_start_by_matchup = _odds_start_times(games + totals_games)
        kalshi_by_series = (
            get_kalshi_markets(
                target_date=target_date,
                tz_name="US/Central",
                date_window_days=DATE_WINDOW_DAYS,
            )
            if ENABLE_MONEYLINE
            else {}
        )
        kalshi_totals = (
            get_kalshi_totals_markets(
                target_date=target_date,
                tz_name="US/Central",
                date_window_days=DATE_WINDOW_DAYS,
            )
            if ENABLE_TOTALS
            else []
        )
        ncaam_markets = len(kalshi_by_series.get("KXNCAAMBGAME", {}))
        ncaaw_markets = len(kalshi_by_series.get("KXNCAAWBGAME", {}))
        odds_m = sum(1 for g in games if g.get("sport_key") == "basketball_ncaab")
        odds_w = sum(1 for g in games if g.get("sport_key") == "basketball_wncaab")
        edges_m = 0
        edges_w = 0
        balance = parse_balance(get_balance())
        cash_available = balance["cash_available"]
        portfolio_value = balance["portfolio_value"]
        bankroll = cash_available if cash_available is not None else None
        if cash_available is not None:
            print(f"[INFO] Cash available: ${cash_available:,.2f}")
        if portfolio_value is not None:
            print(f"[INFO] Portfolio value: ${portfolio_value:,.2f}")
        if bankroll is None:
            print("[WARN] Cash available missing; skipping stake sizing.")
        if not ASSUME_LIMIT_ORDER:
            print("[WARN] ASSUME_LIMIT_ORDER is False; skipping order placement.")
        odds_teams = set()
        odds_matchups = set()
        for g in games:
            odds_teams.add(g["home"])
            odds_teams.add(g["away"])
        for g in totals_games:
            odds_matchups.add(_matchup_key(g["away"], g["home"]))
        odds_books_by_matchup = {}
        for g in games:
            key = _matchup_key(g["away"], g["home"])
            odds_books_by_matchup[key] = g.get("odds_by_book") or {}
        odds_game_by_matchup = {}
        for g in games:
            key = _matchup_key(g["away"], g["home"])
            odds_game_by_matchup[key] = g
        totals_books_by_key = {}
        for g in totals_games:
            key = _matchup_key(g["away"], g["home"])
            totals_books_by_key[(key, g.get("total"))] = g.get("totals_by_book") or {}
        totals_game_by_key = {}
        for g in totals_games:
            key = _matchup_key(g["away"], g["home"])
            totals_game_by_key[(key, g.get("total"))] = g

        kalshi_keys_all = set()
        if ENABLE_MONEYLINE:
            for series_map in kalshi_by_series.values():
                kalshi_keys_all.update(series_map.keys())
            total_kalshi = sum(len(m) for m in kalshi_by_series.values())
            print(f"[INFO] Kalshi teams loaded: {total_kalshi}")
            print(f"[INFO] Odds teams loaded: {len(odds_teams)}")
        if ENABLE_TOTALS:
            print(f"[INFO] Kalshi totals loaded: {len(kalshi_totals)}")
            print(f"[INFO] Odds totals loaded: {len(odds_matchups)}")
        odds_h2h_rem = odds_cache_remaining("h2h")
        odds_tot_rem = odds_cache_remaining("totals") if ENABLE_TOTALS else None
        remaining_list = [r for r in (odds_h2h_rem, odds_tot_rem) if r is not None]
        if remaining_list:
            remaining = min(remaining_list)
            mins, secs = divmod(remaining, 60)
            print(f"[INFO] Odds API refresh in: {mins:02d}m {secs:02d}s")
        print("[INFO] Resting orders check: enabled")
        print("Team | Side | Book odds | Kalshi price | Edge | Volume | Contracts | Exp $ | EV/$")
        print("-" * 80)

        rows = []
        moneyline_rows = []
        totals_rows = []
        moneyline_candidates = []
        matched = set()
        unmatched_moneyline = []
        unmatched_totals = []
        not_entered = []

        if ENABLE_MONEYLINE:
            for g in games:
                sport_key = g.get("sport_key")
                series_expected = _series_for_sport(sport_key)
                if not series_expected:
                    continue
                matchup_key = _matchup_key(g["away"], g["home"])
                kalshi = kalshi_by_series.get(series_expected, {})
                p_home, p_away = _devig_probs(g["odds_home"], g["odds_away"])
                for team, odds_val, book_prob in (
                    (g["home"], g["odds_home"], p_home),
                    (g["away"], g["odds_away"], p_away),
                ):
                    if team == g["home"]:
                        opp_team = g["away"]
                        opp_odds_val = g["odds_away"]
                        opp_book_prob = p_away
                    else:
                        opp_team = g["home"]
                        opp_odds_val = g["odds_home"]
                        opp_book_prob = p_home
                    k = kalshi.get(team)
                    if not k:
                        unmatched_moneyline.append(team)
                        continue

                    market_ticker = k.get("ticker")
                    series_actual = _series_from_ticker(market_ticker)
                    if series_actual and series_actual != series_expected:
                        not_entered.append(f"{team} YES (series mismatch)")
                        continue

                    matched.add(team)
                    raw_market = k.get("raw_market") or {}
                    yes_bid = _norm_price(raw_market.get("yes_bid"))
                    no_bid = _norm_price(raw_market.get("no_bid"))
                    kalshi_yes = yes_bid
                    kalshi_no = no_bid
                    if kalshi_yes is None:
                        kalshi_yes = k.get("yes_price")
                    if kalshi_no is None:
                        kalshi_no = k.get("no_price")
                    volume = k.get("volume") or 0
                    liquidity = k.get("liquidity")
                    open_interest = k.get("open_interest")
                    event_ticker = k.get("event_ticker")
                    if _market_started(k.get("raw_market"), odds_start_by_matchup):
                        continue

                    contracts_available = (
                        int(liquidity)
                        if isinstance(liquidity, (int, float)) and liquidity > 0
                        else int(open_interest)
                        if isinstance(open_interest, (int, float)) and open_interest > 0
                        else 10**9
                    )

                    if (
                        yes_bid is not None
                        and kalshi_yes is not None
                        and kalshi_yes <= MAX_KALSHI_PRICE
                    ):
                        moneyline_candidates.append(
                            {
                                "team": team,
                                "side": "YES",
                                "odds_val": odds_val,
                                "book_prob": book_prob,
                                "kalshi_price": kalshi_yes,
                                "volume": volume,
                                "contracts_available": contracts_available,
                                "market_ticker": market_ticker,
                                "event_ticker": event_ticker,
                                "market_type": "MONEYLINE",
                                "matchup_key": matchup_key,
                            }
                        )
                    if (
                        no_bid is not None
                        and kalshi_no is not None
                        and kalshi_no <= MAX_KALSHI_PRICE
                    ):
                        moneyline_candidates.append(
                            {
                                "team": team,
                                "side": "NO",
                                "odds_val": opp_odds_val,
                                "book_prob": opp_book_prob,
                                "kalshi_price": kalshi_no,
                                "volume": volume,
                                "contracts_available": contracts_available,
                                "market_ticker": market_ticker,
                                "event_ticker": event_ticker,
                                "market_type": "MONEYLINE",
                                "matchup_key": matchup_key,
                            }
                        )

        if USE_CHEAPEST_IMPLIED_WINNER and moneyline_candidates:
            market_map = _market_by_ticker(kalshi_by_series)
            best_by_implied = {}
            keep = []
            for row in moneyline_candidates:
                team = row.get("team")
                side = row.get("side")
                market_ticker = row.get("market_ticker")
                event_ticker = row.get("event_ticker")
                market = market_map.get(market_ticker) if market_ticker else None
                implied = (
                    _implied_winner(normalize_team(team), side, market)
                    if market
                    else None
                )
                if not implied:
                    keep.append(row)
                    continue
                event_key = event_ticker or market_ticker
                key = (event_key, implied)
                cur = best_by_implied.get(key)
                cur_price = cur.get("kalshi_price") if cur else None
                cur_side = cur.get("side") if cur else None
                cand_price = row.get("kalshi_price", 1)
                chosen_price, chosen_side = _choose_implied_candidate(
                    None if cur_price is None else (cur_price, cur_side),
                    (cand_price, side),
                    YES_NO_TIE_PREFERENCE,
                )
                if cur is None or (
                    chosen_price == cand_price and chosen_side == side
                ):
                    best_by_implied[key] = row
            moneyline_candidates = keep + list(best_by_implied.values())

        for row in moneyline_candidates:
            team = row.get("team")
            side = row.get("side")
            odds_val = row.get("odds_val")
            kalshi_price = row.get("kalshi_price")
            book_prob = row.get("book_prob")
            volume = row.get("volume") or 0
            contracts_available = row.get("contracts_available") or 10**9
            market_ticker = row.get("market_ticker")
            event_ticker = row.get("event_ticker")

            if not _moneyline_side_allowed(side):
                continue
            if kalshi_price is None or kalshi_price > MAX_KALSHI_PRICE:
                continue
            implied = american_to_prob(odds_val)
            if not (0.01 <= book_prob <= 0.99):
                not_entered.append(f"{team} {side} (p_true out of bounds)")
                continue
            if abs(implied - book_prob) > 0.35:
                not_entered.append(f"{team} {side} (p_true mismatch)")
                continue
            if volume < MIN_VOLUME:
                not_entered.append(f"{team} {side} (vol<{MIN_VOLUME})")
                continue

            band = _prob_band(book_prob)
            edge = calculate_edge(book_prob, kalshi_price, KALSHI_FEE)
            ev_per_dollar_val = ev_per_dollar(book_prob, kalshi_price)
            contracts = (
                stake_size(
                    bankroll,
                    book_prob,
                    kalshi_price,
                    contracts_available,
                    is_limit_order=ASSUME_LIMIT_ORDER,
                    kelly_frac=KELLY_FRAC,
                    max_pct=MAX_TRADE_PCT,
                )
                if bankroll is not None
                else 0
            )
            if kalshi_price < 0.08:
                contracts = min(contracts, 5)
            if ev_per_dollar_val > 0.25:
                contracts = int(contracts * 0.5)
            exp_profit = (
                expected_profit_per_contract(
                    book_prob, kalshi_price, ASSUME_LIMIT_ORDER
                )
                * contracts
            )
            moneyline_rows.append(
                (
                    team,
                    side,
                    odds_val,
                    kalshi_price,
                    edge,
                    volume,
                    book_prob,
                    contracts,
                    exp_profit,
                    ev_per_dollar_val,
                    market_ticker,
                    event_ticker,
                    band,
                    "MONEYLINE",
                    None,
                    row.get("matchup_key"),
                )
            )

        if ENABLE_TOTALS:
            totals_by_matchup = {}
            for m in kalshi_totals:
                key = m.get("matchup_key")
                if not key:
                    continue
                totals_by_matchup.setdefault(key, []).append(m)

            totals_possible = 0
            for g in totals_games:
                matchup_key = _matchup_key(g["away"], g["home"])
                p_over, p_under = _devig_probs(g["over_odds"], g["under_odds"])
                for side, odds_val, book_prob in (
                    ("OVER", g["over_odds"], p_over),
                    ("UNDER", g["under_odds"], p_under),
                ):
                    market = _find_totals_market(
                        totals_by_matchup.get(matchup_key, []),
                        matchup_key,
                        side,
                        g["total"],
                    )
                    if not market:
                        unmatched_totals.append(f"{matchup_key} {side} {g['total']}")
                        continue
                    if _market_started(market.get("raw_market"), odds_start_by_matchup):
                        continue
                    totals_possible += 1
                    matched.add(f"{matchup_key}:{side}:{g['total']}")

                    raw_market = market.get("raw_market") or {}
                    yes_bid = _norm_price(raw_market.get("yes_bid"))
                    kalshi_price = yes_bid
                    if kalshi_price is None:
                        kalshi_price = market.get("price")
                    if kalshi_price is None or kalshi_price > MAX_KALSHI_PRICE:
                        continue
                    if yes_bid is None:
                        continue
                    volume = market.get("volume") or 0
                    implied = american_to_prob(odds_val)
                    if not (0.01 <= book_prob <= 0.99):
                        not_entered.append(f"{matchup_key} {side} (p_true out of bounds)")
                        continue
                    if abs(implied - book_prob) > 0.35:
                        not_entered.append(f"{matchup_key} {side} (p_true mismatch)")
                        continue
                    if volume < MIN_VOLUME:
                        not_entered.append(f"{matchup_key} {side} (vol<{MIN_VOLUME})")
                        continue
                    market_ticker = market.get("ticker")
                    event_ticker = market.get("event_ticker")
                    band = _prob_band(book_prob)
                    edge = calculate_edge(book_prob, kalshi_price, KALSHI_FEE)
                    ev_per_dollar_val = ev_per_dollar(book_prob, kalshi_price)
                    contracts = (
                        stake_size(
                            bankroll,
                            book_prob,
                            kalshi_price,
                            10**9,
                            is_limit_order=ASSUME_LIMIT_ORDER,
                            kelly_frac=KELLY_FRAC,
                            max_pct=MAX_TRADE_PCT,
                        )
                        if bankroll is not None
                        else 0
                    )
                    if kalshi_price < 0.08:
                        contracts = min(contracts, 5)
                    if ev_per_dollar_val > 0.25:
                        contracts = int(contracts * 0.5)
                    exp_profit = (
                        expected_profit_per_contract(
                            book_prob, kalshi_price, ASSUME_LIMIT_ORDER
                        )
                        * contracts
                    )
                    label = f"{g['away']} at {g['home']} total {g['total']} {side}"
                    totals_rows.append(
                        (
                            label,
                            "YES",
                            odds_val,
                            kalshi_price,
                            edge,
                            volume,
                            book_prob,
                            contracts,
                            exp_profit,
                            ev_per_dollar_val,
                            market_ticker,
                            event_ticker,
                            band,
                            "TOTAL",
                            g["total"],
                            matchup_key,
                        )
                    )

        rows = moneyline_rows + totals_rows

        if not rows:
            print("[INFO] No matched teams found.")
        else:
            rows.sort(key=lambda r: r[4], reverse=True)
            expected_rows = 0
            if ENABLE_MONEYLINE:
                expected_rows += len(moneyline_candidates)
            if ENABLE_TOTALS:
                expected_rows += totals_possible
            print(f"[INFO] Matched rows: {len(rows)} / {expected_rows}")
            edges_found = []
            for (
                team,
                side,
                odds_val,
                kalshi_price,
                edge,
                volume,
                book_prob,
                contracts,
                exp_profit,
                ev_per_dollar_val,
                market_ticker,
                event_ticker,
                band,
                market_type,
                total_line,
                matchup_key,
            ) in rows:
                if edge <= 0 or contracts <= 0:
                    continue
                min_edge_dyn = _min_edge_for_prob(book_prob)
                if edge >= min_edge_dyn:
                    print(
                        f"{team} | {side} | {_format_odds(odds_val)} | p={book_prob:.3f} {band} | "
                        f"{kalshi_price:.3f} | {edge*100:.2f}% | {volume} | {contracts} | "
                        f"${exp_profit:,.2f} | {ev_per_dollar_val:.4f}"
                    )
                    edges_found.append(
                        {
                            "team": team,
                            "side": side,
                            "book_odds": odds_val,
                            "p_true": round(book_prob, 6),
                            "prob_band": band,
                            "kalshi_price": round(kalshi_price, 6),
                            "edge": round(edge, 6),
                            "volume": volume,
                            "contracts": contracts,
                            "expected_profit": round(exp_profit, 2),
                            "ev_per_dollar": round(ev_per_dollar_val, 6),
                            "market_ticker": market_ticker,
                            "event_ticker": event_ticker,
                            "market_type": market_type,
                            "total_line": total_line,
                            "matchup_key": matchup_key,
                        }
                    )
                    if QUIET_LOGS:
                        _EDGE_COUNTS["total"] += 1
                        if edge >= 0.05:
                            _EDGE_COUNTS["ge_5"] += 1
                        if edge >= 0.03:
                            _EDGE_COUNTS["ge_3"] += 1
                        if edge >= 0.02:
                            _EDGE_COUNTS["ge_2"] += 1
                        if edge >= 0.01:
                            _EDGE_COUNTS["ge_1"] += 1
                    else:
                        alert_edge(team, kalshi_price, book_prob, edge)
                else:
                    reason = []
                    if edge < min_edge_dyn:
                        reason.append(f"edge<{min_edge_dyn:.3f}")
                    not_entered.append(
                        f"{team} {side} ({', '.join(reason) if reason else 'rule'})"
                    )
            if not QUIET_LOGS:
                print(f"[INFO] Unmatched moneyline teams: {len(unmatched_moneyline)}")
            if unmatched_moneyline:
                # Write full unmatched list + closest Kalshi keys for fast mapping
                unmatched_path = "data/unmatched_teams.csv"
                kalshi_keys = sorted(kalshi_keys_all)
                with open(unmatched_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "odds_team",
                            "normalized",
                            "closest_kalshi_1",
                            "closest_kalshi_2",
                            "closest_kalshi_3",
                        ]
                    )
                    for t in sorted(set(unmatched_moneyline)):
                        matches = difflib.get_close_matches(t, kalshi_keys, n=3, cutoff=0.6)
                        while len(matches) < 3:
                            matches.append("")
                        writer.writerow([t, t, matches[0], matches[1], matches[2]])

            if ENABLE_TOTALS:
                if not QUIET_LOGS:
                    print(f"[INFO] Unmatched totals: {len(unmatched_totals)}")
                if unmatched_totals:
                    unmatched_totals_path = "data/unmatched_totals.csv"
                    with open(unmatched_totals_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["matchup_key"])
                        for t in sorted(set(unmatched_totals)):
                            writer.writerow([t])
            if not_entered:
                print(f"[INFO] Found but not entered: {len(not_entered)}")
                for msg in not_entered[:20]:
                    print(f"[SKIP] {msg}")

            if USE_CHEAPEST_IMPLIED_WINNER and edges_found:
                market_map = _market_by_ticker(kalshi_by_series)
                keep = []
                best_by_implied = {}
                for row in edges_found:
                    if row.get("market_type") != "MONEYLINE":
                        keep.append(row)
                        continue
                    ticker = row.get("market_ticker")
                    market = market_map.get(ticker) if ticker else None
                    implied = (
                        _implied_winner(row.get("team"), row.get("side"), market)
                        if market
                        else None
                    )
                    if not implied:
                        keep.append(row)
                        continue
                    event_key = row.get("event_ticker") or ticker
                    key = (event_key, implied)
                    cur = best_by_implied.get(key)
                    cur_price = cur.get("kalshi_price") if cur else None
                    cur_side = cur.get("side") if cur else None
                    cand_price = row.get("kalshi_price", 1)
                    cand_side = row.get("side")
                    bid_price = _best_bid_for_row(row, market) or cand_price
                    chosen_price, chosen_side = _choose_implied_candidate(
                        None if cur_price is None else (cur_price, cur_side),
                        (bid_price, cand_side),
                        YES_NO_TIE_PREFERENCE,
                    )
                    if cur is None or (chosen_price == bid_price and chosen_side == cand_side):
                        best_by_implied[key] = row
                edges_found = keep + list(best_by_implied.values())

            edges_m = sum(
                1
                for r in edges_found
                if r.get("market_type") == "MONEYLINE"
                and _series_from_ticker(r.get("market_ticker")) == "KXNCAAMBGAME"
            )
            edges_w = sum(
                1
                for r in edges_found
                if r.get("market_type") == "MONEYLINE"
                and _series_from_ticker(r.get("market_ticker")) == "KXNCAAWBGAME"
            )

            csv_path = "data/edges_found.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "team",
                        "side",
                        "book_odds",
                        "p_true",
                        "prob_band",
                        "kalshi_price",
                        "edge",
                        "volume",
                        "contracts",
                        "expected_profit",
                        "ev_per_dollar",
                        "market_ticker",
                        "event_ticker",
                        "market_type",
                        "total_line",
                        "matchup_key",
                    ],
                )
                writer.writeheader()
                writer.writerows(edges_found)
            print(f"[INFO] Wrote {len(edges_found)} edges to {csv_path}")

            ledger = load_ledger()
            market_map = _market_by_ticker(kalshi_by_series)
            for m in kalshi_totals:
                t = m.get("ticker")
                if t and m.get("raw_market"):
                    market_map[t] = m.get("raw_market")
            market_cache = dict(market_map)
            ticker_to_team = _ticker_to_team(kalshi_by_series)
            event_team_yes = _event_team_yes_prices(kalshi_by_series)
            positions = get_positions()
            orders = {"orders": []}
            canceled_ids = set()
            if OPEN_ORDERS_STATUS:
                statuses = [s.strip() for s in OPEN_ORDERS_STATUS.split(",") if s.strip()]
                for status in statuses:
                    resp = get_orders(params={"status": status})
                    batch = resp.get("orders") or resp.get("data") or []
                    orders["orders"].extend(batch)
            else:
                orders = get_orders()
            if TRADE_MODE == "live":
                try:
                    canceled = get_orders(params={"status": "canceled"})
                    canceled_batch = canceled.get("orders") or canceled.get("data") or []
                    for o in canceled_batch:
                        oid = o.get("order_id") or o.get("id")
                        if oid:
                            canceled_ids.add(str(oid))
                except Exception:
                    canceled_ids = set()
            # Save sorted snapshots for inspection
            if isinstance(positions, dict):
                positions["market_positions"] = _sorted_snapshot_list(
                    positions.get("market_positions") or []
                )
                positions["event_positions"] = _sorted_snapshot_list(
                    positions.get("event_positions") or []
                )
            if isinstance(orders, dict):
                orders["orders"] = _sorted_snapshot_list(orders.get("orders") or [])
            save_positions_snapshot(positions)
            save_resting_orders_snapshot(orders)
            positions_count = len(
                positions.get("market_positions")
                or positions.get("positions")
                or positions.get("data")
                or []
            )
            open_orders = _open_orders_map(orders)
            print(f"[INFO] Positions loaded: {positions_count}")
            print(f"[INFO] Resting orders loaded: {len(open_orders)}")
            resting_state = load_resting_state()
            positions_exposure = _positions_exposure(positions, market_map, ticker_to_team)
            positions_tickers = _positions_tickers(positions)
            positions_by_market = _positions_by_market(positions)
            positions_by_event = _positions_by_event(positions, market_map)
            orders_exposure = _orders_exposure(orders, market_map, ticker_to_team)
            print(f"[INFO] Resting order audit: {len(open_orders)} open orders")
            orders_submitted = 0
            event_implied = {}
            for src in (positions_exposure, orders_exposure):
                for k, v in _event_implied_map(src, market_map).items():
                    event_implied.setdefault(k, set()).update(v)

            open_order_dollars_by_market = {}
            open_order_dollars_by_event = {}
            for (ticker, _side), info in open_orders.items():
                price = info.get("price") or 0
                remaining = info.get("remaining") or 0
                try:
                    price = float(price)
                    remaining = float(remaining)
                except Exception:
                    continue
                if price <= 0 or remaining <= 0:
                    continue
                notional = price * remaining
                open_order_dollars_by_market[ticker] = open_order_dollars_by_market.get(ticker, 0) + notional
                market = market_map.get(ticker) or {}
                event = market.get("event_ticker") or ticker
                open_order_dollars_by_event[event] = open_order_dollars_by_event.get(event, 0) + notional

            best_price_by_implied = {}
            if USE_CHEAPEST_IMPLIED_WINNER:
                for row in edges_found:
                    if row.get("market_type") != "MONEYLINE":
                        continue
                    ticker = row.get("market_ticker")
                    market = market_map.get(ticker) if ticker else None
                    implied = (
                        _implied_winner(row.get("team"), row.get("side"), market)
                        if market
                        else None
                    )
                    if not implied:
                        continue
                    event_key = row.get("event_ticker") or ticker
                    key = (event_key, implied)
                    price = row.get("kalshi_price")
                    side = row.get("side")
                    bid_price = _best_bid_for_row(row, market) or price
                    cur = best_price_by_implied.get(key)
                    cur_price = cur["price"] if cur else None
                    cur_side = cur["side"] if cur else None
                    chosen_price, chosen_side = _choose_implied_candidate(
                        None if cur_price is None else (cur_price, cur_side),
                        (bid_price, side),
                        YES_NO_TIE_PREFERENCE,
                    )
                    if cur is None or (chosen_price == bid_price and chosen_side == side):
                        best_price_by_implied[key] = {"price": bid_price, "side": side}

            # Report arbs based on current positions + resting orders
            holdings = []
            holdings.extend(_holdings_from_positions(positions, market_map, ticker_to_team))
            holdings.extend(_holdings_from_orders(orders, market_map, ticker_to_team))
            _report_arbs(holdings, market_map)

            # Cancel open orders that no longer meet edge threshold
            odds_map_by_series = {}
            odds_raw_by_series = {}
            for g in games:
                series_expected = _series_for_sport(g.get("sport_key"))
                if not series_expected:
                    continue
                p_home, p_away = _devig_probs(g["odds_home"], g["odds_away"])
                odds_map_by_series.setdefault(series_expected, {})
                odds_map_by_series[series_expected][g["home"]] = p_home
                odds_map_by_series[series_expected][g["away"]] = p_away
                odds_raw_by_series.setdefault(series_expected, {})
                odds_raw_by_series[series_expected][g["home"]] = g.get("odds_home")
                odds_raw_by_series[series_expected][g["away"]] = g.get("odds_away")

            totals_prob_by_ticker = {}
            if ENABLE_TOTALS:
                totals_by_matchup = {}
                for m in kalshi_totals:
                    key = m.get("matchup_key")
                    if not key:
                        continue
                    totals_by_matchup.setdefault(key, []).append(m)
                for g in totals_games:
                    matchup_key = _matchup_key(g["away"], g["home"])
                    p_over, p_under = _devig_probs(g["over_odds"], g["under_odds"])
                    for side, book_prob in (("OVER", p_over), ("UNDER", p_under)):
                        market = _find_totals_market(
                            totals_by_matchup.get(matchup_key, []),
                            matchup_key,
                            side,
                            g["total"],
                        )
                        if market and market.get("ticker"):
                            totals_prob_by_ticker[market["ticker"]] = book_prob
            for (ticker, side), info in open_orders.items():
                book_prob = None
                team = ticker_to_team.get(ticker)
                series_actual = _series_from_ticker(ticker)
                market = market_cache.get(ticker)
                if market is None:
                    fetched = get_market_by_ticker(ticker)
                    if fetched:
                        market_cache[ticker] = fetched
                        market = fetched

                lookup_team = normalize_team(team) if team else None
                if team and series_actual in odds_map_by_series:
                    if side == "NO" and market:
                        implied = _implied_winner(team, side, market)
                        if implied:
                            lookup_team = implied
                    book_prob = odds_map_by_series[series_actual].get(lookup_team)
                elif ticker in totals_prob_by_ticker:
                    book_prob = totals_prob_by_ticker[ticker]
                if book_prob is None:
                    continue
                if side == "NO" and team and lookup_team == normalize_team(team):
                    book_prob = 1 - book_prob
                price = info["price"]
                mid_price = _mid_price(market, side) if market else None
                ev_mid = book_prob - (mid_price if mid_price is not None else price)
                current_odds = None
                if lookup_team and series_actual in odds_raw_by_series:
                    current_odds = odds_raw_by_series[series_actual].get(lookup_team)
                if current_odds is None:
                    current_odds = _prob_to_american(book_prob)
                old_odds = None
                order_meta = resting_state.get("order_book_odds", {})
                order_id = info.get("order_id")
                if order_id and isinstance(order_meta, dict):
                    old_odds = order_meta.get(str(order_id))
                odds_info = ""
                if old_odds is not None and current_odds is not None:
                    odds_info = (
                        f" odds {_format_odds(old_odds)} -> {_format_odds(current_odds)}"
                    )
                if _market_started(market, odds_start_by_matchup):
                    if TRADE_MODE == "live":
                        try:
                            cancel_order(info["order_id"])
                            if info.get("order_id"):
                                canceled_ids.add(str(info["order_id"]))
                            print(f"[INFO] Canceled resting (game started): {ticker} {side}")
                        except Exception as exc:
                            if "404" in str(exc):
                                print(f"[INFO] Cancel not found (already gone): {ticker} {side}")
                            else:
                                print(f"[WARN] Cancel failed: {ticker} {side} -> {exc}")
                    continue
                elif market:
                    away, home = _matchup_teams(market)
                    if away and home:
                        key = _matchup_key(away, home)
                        if key not in odds_start_by_matchup:
                            print(f"[INFO] No Odds API start time for: {ticker} {side} ({key})")
                best_bid = _post_only_price(market, side)
                off_market = (
                    best_bid is not None
                    and (price - best_bid) >= OFF_MARKET_CENTS
                    and ev_mid <= CANCEL_EV_BUFFER
                )
                not_competitive = (
                    best_bid is not None
                    and price is not None
                    and (best_bid - price) >= CANCEL_NOT_COMPETITIVE_GAP
                    and ev_mid < CANCEL_NOT_COMPETITIVE_EV
                )
                wait_key = f"ev_below:{ticker}:{side}"
                off_wait_key = f"off_market:{ticker}:{side}"
                nc_wait_key = f"not_competitive:{ticker}:{side}"
                now_ts = time.time()
                if ev_mid < CANCEL_EV_BUFFER:
                    first_ts = resting_state.get(wait_key)
                    if first_ts is None:
                        resting_state[wait_key] = now_ts
                        save_resting_state(resting_state)
                        first_ts = now_ts
                    if CANCEL_TIME_SEC and (now_ts - first_ts) < CANCEL_TIME_SEC:
                        print(
                            f"[SKIP] cancel (buffer): {ticker} {side} "
                            f"ev_mid={ev_mid:.3f} mid={mid_price if mid_price is not None else 'n/a'}"
                        )
                        continue
                else:
                    if wait_key in resting_state:
                        resting_state.pop(wait_key, None)
                        save_resting_state(resting_state)
                if off_market:
                    first_ts = resting_state.get(off_wait_key)
                    if first_ts is None:
                        resting_state[off_wait_key] = now_ts
                        save_resting_state(resting_state)
                        first_ts = now_ts
                    if OFF_MARKET_TIME_SEC and (now_ts - first_ts) < OFF_MARKET_TIME_SEC:
                        print(
                            f"[SKIP] cancel (off-market buffer): {ticker} {side} "
                            f"off_by={(price - best_bid):.2f} bid={best_bid:.2f}"
                        )
                        continue
                else:
                    if off_wait_key in resting_state:
                        resting_state.pop(off_wait_key, None)
                        save_resting_state(resting_state)
                if not_competitive:
                    first_ts = resting_state.get(nc_wait_key)
                    if first_ts is None:
                        resting_state[nc_wait_key] = now_ts
                        save_resting_state(resting_state)
                        first_ts = now_ts
                    if CANCEL_NOT_COMPETITIVE_TIME_SEC and (
                        now_ts - first_ts
                    ) < CANCEL_NOT_COMPETITIVE_TIME_SEC:
                        print(
                            f"[SKIP] cancel (not-competitive buffer): {ticker} {side} "
                            f"gap={(best_bid - price):.2f} bid={best_bid:.2f}"
                        )
                        continue
                else:
                    if nc_wait_key in resting_state:
                        resting_state.pop(nc_wait_key, None)
                        save_resting_state(resting_state)

                if ev_mid < CANCEL_EV_BUFFER or off_market or not_competitive:
                    if TRADE_MODE == "live":
                        try:
                            cancel_order(info["order_id"])
                            if info.get("order_id"):
                                canceled_ids.add(str(info["order_id"]))
                            if ev_mid < CANCEL_EV_BUFFER:
                                reason = "ev"
                            elif off_market:
                                reason = "off-market"
                            else:
                                reason = "not-competitive"
                            print(
                                f"[INFO] Canceled resting ({reason}): {ticker} {side} "
                                f"ev_mid={ev_mid:.3f} bid={best_bid if best_bid is not None else 'n/a'}{odds_info}"
                            )
                        except Exception as exc:
                            if "404" in str(exc):
                                print(f"[INFO] Cancel not found (already gone): {ticker} {side}")
                            else:
                                print(f"[WARN] Cancel failed: {ticker} {side} -> {exc}")
                    if wait_key in resting_state:
                        resting_state.pop(wait_key, None)
                        save_resting_state(resting_state)
                    if off_wait_key in resting_state:
                        resting_state.pop(off_wait_key, None)
                        save_resting_state(resting_state)
                    if nc_wait_key in resting_state:
                        resting_state.pop(nc_wait_key, None)
                        save_resting_state(resting_state)
            # Place simple in-market arbs using best bid on both sides
            if ENABLE_ARBS:
                arb_orders_submitted = 0
                arb_markets = _arb_candidates(market_map)
                for arb in arb_markets:
                    if arb_orders_submitted >= ARB_MAX_ORDERS_PER_RUN:
                        break
                    ticker = arb["ticker"]
                    yes_ask = arb["yes_ask"]
                    no_ask = arb["no_ask"]
                    fee_yes = arb.get("fee_yes", 0.0)
                    fee_no = arb.get("fee_no", 0.0)
                    total = arb["total"]
                    profit = arb["profit"]
                    if (ticker, "YES") in open_orders or (ticker, "NO") in open_orders:
                        continue
                    existing_pos = positions_by_market.get(ticker)
                    if existing_pos and existing_pos.get("count", 0) > 0:
                        continue
                    market_dollars = open_order_dollars_by_market.get(ticker, 0.0)
                    if existing_pos and existing_pos.get("count", 0) > 0:
                        market_dollars += (existing_pos.get("avg_price") or 0) * existing_pos.get("count", 0)
                    remaining_dollars = MAX_DOLLARS_PER_MARKET - market_dollars
                    if remaining_dollars <= 0:
                        continue
                    per_contract_cost = yes_ask + no_ask + fee_yes + fee_no
                    if per_contract_cost <= 0:
                        continue
                    contracts = int(min(ARB_MAX_CONTRACTS, remaining_dollars / per_contract_cost))
                    if contracts <= 0:
                        continue
                    payload_yes = {
                        "ticker": ticker,
                        "action": "buy",
                        "side": "yes",
                        "type": "limit",
                        "count": contracts,
                        "client_order_id": str(uuid.uuid4()),
                        "post_only": False,
                        "yes_price_dollars": _price_to_dollars(round(yes_ask, 2)),
                    }
                    payload_no = {
                        "ticker": ticker,
                        "action": "buy",
                        "side": "no",
                        "type": "limit",
                        "count": contracts,
                        "client_order_id": str(uuid.uuid4()),
                        "post_only": False,
                        "no_price_dollars": _price_to_dollars(round(no_ask, 2)),
                    }
                    if TRADE_MODE == "live":
                        try:
                            place_order(payload_yes)
                            place_order(payload_no)
                            print(
                                f"[INFO] Placed ARB: {ticker} YES@{yes_ask:.2f} "
                                f"+ NO@{no_ask:.2f} x{contracts} | sum {total:.2f} "
                                f"| profit {profit:.2f} (fees {fee_yes+fee_no:.2f})"
                            )
                            arb_orders_submitted += 2
                        except Exception as exc:
                            print(f"[WARN] ARB failed: {ticker} -> {exc}")
                    else:
                        print(
                            f"[INFO] DRY RUN ARB: {ticker} YES@{yes_ask:.2f} "
                            f"+ NO@{no_ask:.2f} x{contracts} | sum {total:.2f} "
                            f"| profit {profit:.2f} (fees {fee_yes+fee_no:.2f})"
                        )
                        arb_orders_submitted += 2
            def _format_books_info(row: dict) -> str:
                market_type = row.get("market_type")
                if market_type == "MONEYLINE":
                    matchup_key = row.get("matchup_key")
                    if not matchup_key:
                        return ""
                    books = odds_books_by_matchup.get(matchup_key) or {}
                    if not books:
                        return ""
                    team = normalize_team(row.get("team") or "")
                    if not team:
                        return ""
                    opponent = None
                    for book_odds in books.values():
                        if team in book_odds:
                            for name in book_odds:
                                if name != team:
                                    opponent = name
                                    break
                        if opponent:
                            break
                    parts = []
                    for book_key, book_odds in books.items():
                        team_odds = book_odds.get(team)
                        opp_odds = book_odds.get(opponent) if opponent else None
                        if team_odds is None and opp_odds is None:
                            continue
                        team_str = _format_odds(team_odds) if team_odds is not None else "n/a"
                        if opp_odds is None:
                            parts.append(f"{book_key}:{team_str}")
                        else:
                            opp_str = _format_odds(opp_odds) if opp_odds is not None else "n/a"
                            parts.append(f"{book_key}:{team_str}/{opp_str}")
                    return f" | books {', '.join(parts)}" if parts else ""
                if market_type == "TOTAL":
                    matchup_key = row.get("matchup_key")
                    total_line = row.get("total_line")
                    if matchup_key is None or total_line is None:
                        return ""
                    books = totals_books_by_key.get((matchup_key, total_line)) or {}
                    if not books:
                        return ""
                    parts = []
                    for book_key, sides in books.items():
                        over = sides.get("over")
                        under = sides.get("under")
                        if over is None and under is None:
                            continue
                        over_str = _format_odds(over) if over is not None else "n/a"
                        under_str = _format_odds(under) if under is not None else "n/a"
                        parts.append(f"{book_key}:O{over_str}/U{under_str}")
                    return f" | books {', '.join(parts)}" if parts else ""
                return ""

            def _format_math_info(row: dict) -> str:
                market_type = row.get("market_type")
                if market_type == "MONEYLINE":
                    matchup_key = row.get("matchup_key")
                    if not matchup_key:
                        return ""
                    game = odds_game_by_matchup.get(matchup_key)
                    if not game:
                        return ""
                    home = game.get("home")
                    away = game.get("away")
                    odds_home = game.get("odds_home")
                    odds_away = game.get("odds_away")
                    try:
                        p_home_raw = american_to_prob(float(odds_home))
                        p_away_raw = american_to_prob(float(odds_away))
                    except Exception:
                        return ""
                    p_home_dev, p_away_dev = _devig_probs(odds_home, odds_away)
                    team = normalize_team(row.get("team") or "")
                    side = row.get("side")
                    opp = away if team == home else home
                    used_team = team if side == "YES" else opp
                    odds_home_str = _format_odds(odds_home)
                    odds_away_str = _format_odds(odds_away)
                    if used_team == home:
                        p_used = p_home_dev
                    elif used_team == away:
                        p_used = p_away_dev
                    else:
                        p_used = None
                    used_str = f" used={p_used:.3f}" if p_used is not None else ""
                    return (
                        f" | math odds H/A={odds_home_str}/{odds_away_str} "
                        f"raw H/A={p_home_raw:.3f}/{p_away_raw:.3f} "
                        f"devig {home}={p_home_dev:.3f} {away}={p_away_dev:.3f} "
                        f"used_team={used_team}{used_str}"
                    )
                if market_type == "TOTAL":
                    matchup_key = row.get("matchup_key")
                    total_line = row.get("total_line")
                    if matchup_key is None or total_line is None:
                        return ""
                    game = totals_game_by_key.get((matchup_key, total_line))
                    if not game:
                        return ""
                    odds_over = game.get("over_odds")
                    odds_under = game.get("under_odds")
                    try:
                        p_over_raw = american_to_prob(float(odds_over))
                        p_under_raw = american_to_prob(float(odds_under))
                    except Exception:
                        return ""
                    p_over_dev, p_under_dev = _devig_probs(odds_over, odds_under)
                    label = row.get("team") or ""
                    side_hint = "OVER" if label.endswith(" OVER") else "UNDER" if label.endswith(" UNDER") else None
                    p_used = p_over_dev if side_hint == "OVER" else p_under_dev if side_hint == "UNDER" else None
                    used_str = f" used={p_used:.3f}" if p_used is not None else ""
                    odds_over_str = _format_odds(odds_over)
                    odds_under_str = _format_odds(odds_under)
                    return (
                        f" | math odds O/U={odds_over_str}/{odds_under_str} "
                        f"raw O/U={p_over_raw:.3f}/{p_under_raw:.3f} "
                        f"devig O={p_over_dev:.3f} U={p_under_dev:.3f} "
                        f"used_side={side_hint}{used_str}"
                    )
                return ""

            for row in edges_found:
                if orders_submitted >= MAX_ORDERS_PER_RUN:
                    break
                market_ticker = row.get("market_ticker")
                side = row.get("side")
                event_key = row.get("event_ticker") or market_ticker
                has_resting = (market_ticker, side) in open_orders
                price = row.get("kalshi_price")
                contracts = row.get("contracts", 0)
                if not market_ticker:
                    print(f"[SKIP] missing market ticker: {row.get('team')} {side}")
                    continue
                if contracts <= 0:
                    print(f"[SKIP] zero contracts: {market_ticker} {side}")
                    continue
                key = _order_key(market_ticker, side, target_date_str)

                market = market_cache.get(market_ticker)
                if market is None:
                    fetched = get_market_by_ticker(market_ticker)
                    if fetched:
                        market_cache[market_ticker] = fetched
                        market = fetched
                if _market_started(market, odds_start_by_matchup):
                    if has_resting and TRADE_MODE == "live":
                        try:
                            cancel_order(open_orders[(market_ticker, side)]["order_id"])
                            canceled_ids.add(
                                str(open_orders[(market_ticker, side)]["order_id"])
                            )
                            print(f"[INFO] Canceled resting (game started): {market_ticker} {side}")
                        except Exception as exc:
                            if "404" in str(exc):
                                print(f"[INFO] Cancel not found (already gone): {market_ticker} {side}")
                            else:
                                print(f"[WARN] Cancel failed: {market_ticker} {side} -> {exc}")
                    print(f"[SKIP] game started: {market_ticker} {side}")
                    continue
                best_bid = _post_only_price(market, side)
                if best_bid is None:
                    print(f"[SKIP] no bid price: {market_ticker} {side}")
                    continue
                best_bid = round(max(0.01, min(0.99, best_bid)), 2)

                implied_winner = _implied_winner(row.get("team"), side, market)
                if row.get("market_type") == "MONEYLINE" and not implied_winner:
                    print(f"[SKIP] missing implied winner: {market_ticker} {side}")
                    continue
                exposure_key = (market_ticker, implied_winner, side)
                if exposure_key in orders_exposure and not has_resting:
                    print(f"[SKIP] resting exposure (other order): {exposure_key}")
                    continue
                existing_set = event_implied.get(event_key, set())
                if implied_winner:
                    if USE_CHEAPEST_IMPLIED_WINNER:
                        key = (event_key, implied_winner)
                        best = best_price_by_implied.get(key) or {}
                        best_price = best.get("price")
                        best_side = best.get("side")
                        if not _is_preferred_implied(
                            best_bid, side, best_price, best_side, YES_NO_TIE_PREFERENCE
                        ):
                            print(
                                f"[SKIP] not cheapest implied winner: {market_ticker} {side} "
                                f"price={best_bid:.2f} best={best_price:.2f}"
                            )
                            continue
                    if implied_winner in existing_set:
                        print(
                            f"[SKIP] duplicate implied winner: event={event_key} "
                            f"implied_winner={implied_winner} from {side} {row.get('team')}"
                        )
                        continue
                    if not ALLOW_TRUE_HEDGE and existing_set:
                        print(
                            f"[SKIP] hedge blocked: event={event_key} "
                            f"implied_winner={implied_winner} from {side} {row.get('team')}"
                        )
                        continue
                    event_implied.setdefault(event_key, set()).add(implied_winner)

                # EV check using best bid (per your rule)
                true_prob = row.get("p_true")
                if true_prob is None:
                    true_prob = row.get("book_prob")
                if true_prob is None:
                    print(f"[SKIP] missing p_true: {market_ticker} {side}")
                    continue
                book_odds_val = row.get("book_odds")
                p_book = None
                if book_odds_val is not None:
                    try:
                        p_book = american_to_prob(float(book_odds_val))
                    except Exception:
                        p_book = None
                if p_book is not None and abs(true_prob - p_book) > MAX_PROB_GAP:
                    print(
                        f"[SKIP] prob mismatch: {market_ticker} {side} "
                        f"p_true={true_prob:.3f} p_book={p_book:.3f} gap={abs(true_prob - p_book):.3f}"
                    )
                    continue
                edge_at_bid = calculate_edge(true_prob, best_bid, KALSHI_FEE)
                if edge_at_bid < MIN_EDGE:
                    print(
                        f"[SKIP] edge<{MIN_EDGE:.3f} at bid: {market_ticker} {side} "
                        f"edge={edge_at_bid:.3f} bid={best_bid:.2f}"
                    )
                    continue
                ev = true_prob - best_bid
                if ev <= 0:
                    print(f"[SKIP] EV<=0 at bid: {market_ticker} {side}")
                    if has_resting and TRADE_MODE == "live":
                        try:
                            cancel_order(open_orders[(market_ticker, side)]["order_id"])
                            canceled_ids.add(
                                str(open_orders[(market_ticker, side)]["order_id"])
                            )
                            print(f"[INFO] Canceled (EV<=0): {market_ticker} {side}")
                        except Exception as exc:
                            if "404" in str(exc):
                                print(f"[INFO] Cancel not found (already gone): {market_ticker} {side}")
                            else:
                                print(f"[WARN] Cancel failed: {market_ticker} {side} -> {exc}")
                    continue
                price = best_bid

                # Resting order management: cancel + replace if improved by >= $0.01
                if has_resting:
                    rest_key = f"{market_ticker}:{side}"
                    state = resting_state.get(rest_key, {})
                    last_ts = state.get("last_update_ts", 0)
                    updates = state.get("updates", 0)
                    current_price = open_orders[(market_ticker, side)]["price"]

                    # Enforce minimum edge on resting orders; cancel if it no longer meets threshold
                    best_bid = _post_only_price(market, side)
                    mid_price = _mid_price(market, side)
                    ev_mid = true_prob - (mid_price if mid_price is not None else current_price)
                    off_market = (
                        best_bid is not None
                        and (current_price - best_bid) >= OFF_MARKET_CENTS
                        and ev_mid <= CANCEL_EV_BUFFER
                    )
                    not_competitive = (
                        best_bid is not None
                        and (best_bid - current_price) >= CANCEL_NOT_COMPETITIVE_GAP
                        and ev_mid < CANCEL_NOT_COMPETITIVE_EV
                    )
                    wait_key = f"ev_below:{market_ticker}:{side}"
                    off_wait_key = f"off_market:{market_ticker}:{side}"
                    nc_wait_key = f"not_competitive:{market_ticker}:{side}"
                    now_ts = time.time()
                    if ev_mid < CANCEL_EV_BUFFER:
                        first_ts = resting_state.get(wait_key)
                        if first_ts is None:
                            resting_state[wait_key] = now_ts
                            save_resting_state(resting_state)
                            first_ts = now_ts
                        if CANCEL_TIME_SEC and (now_ts - first_ts) < CANCEL_TIME_SEC:
                            print(
                                f"[SKIP] cancel (buffer): {market_ticker} {side} "
                                f"ev_mid={ev_mid:.3f} mid={mid_price if mid_price is not None else 'n/a'}"
                            )
                            continue
                    else:
                        if wait_key in resting_state:
                            resting_state.pop(wait_key, None)
                            save_resting_state(resting_state)
                    if off_market:
                        first_ts = resting_state.get(off_wait_key)
                        if first_ts is None:
                            resting_state[off_wait_key] = now_ts
                            save_resting_state(resting_state)
                            first_ts = now_ts
                        if OFF_MARKET_TIME_SEC and (now_ts - first_ts) < OFF_MARKET_TIME_SEC:
                            print(
                                f"[SKIP] cancel (off-market buffer): {market_ticker} {side} "
                                f"off_by={(current_price - best_bid):.2f} bid={best_bid:.2f}"
                            )
                            continue
                    else:
                        if off_wait_key in resting_state:
                            resting_state.pop(off_wait_key, None)
                            save_resting_state(resting_state)
                    if not_competitive:
                        first_ts = resting_state.get(nc_wait_key)
                        if first_ts is None:
                            resting_state[nc_wait_key] = now_ts
                            save_resting_state(resting_state)
                            first_ts = now_ts
                        if CANCEL_NOT_COMPETITIVE_TIME_SEC and (
                            now_ts - first_ts
                        ) < CANCEL_NOT_COMPETITIVE_TIME_SEC:
                            print(
                                f"[SKIP] cancel (not-competitive buffer): {market_ticker} {side} "
                                f"gap={(best_bid - current_price):.2f} bid={best_bid:.2f}"
                            )
                            continue
                    else:
                        if nc_wait_key in resting_state:
                            resting_state.pop(nc_wait_key, None)
                            save_resting_state(resting_state)

                    if ev_mid < CANCEL_EV_BUFFER or off_market or not_competitive:
                        if TRADE_MODE == "live":
                            try:
                                cancel_order(open_orders[(market_ticker, side)]["order_id"])
                                canceled_ids.add(
                                    str(open_orders[(market_ticker, side)]["order_id"])
                                )
                                if ev_mid < CANCEL_EV_BUFFER:
                                    reason = "ev"
                                elif off_market:
                                    reason = "off-market"
                                else:
                                    reason = "not-competitive"
                                print(
                                    f"[INFO] Canceled ({reason}): {market_ticker} {side} "
                                    f"ev_mid={ev_mid:.3f} bid={best_bid if best_bid is not None else 'n/a'}"
                                )
                            except Exception as exc:
                                if "404" in str(exc):
                                    print(f"[INFO] Cancel not found (already gone): {market_ticker} {side}")
                                else:
                                    print(f"[WARN] Cancel failed: {market_ticker} {side} -> {exc}")
                        if wait_key in resting_state:
                            resting_state.pop(wait_key, None)
                            save_resting_state(resting_state)
                        if off_wait_key in resting_state:
                            resting_state.pop(off_wait_key, None)
                            save_resting_state(resting_state)
                        if nc_wait_key in resting_state:
                            resting_state.pop(nc_wait_key, None)
                            save_resting_state(resting_state)
                        continue

                    if (time.time() - last_ts) < ORDER_UPDATE_COOLDOWN_SEC:
                        print(f"[SKIP] update cooldown: {market_ticker} {side}")
                        continue
                    if updates >= MAX_UPDATES_PER_MARKET:
                        print(f"[SKIP] max updates reached: {market_ticker} {side}")
                        continue
                    # For both YES/NO, better means lower price (pay less).
                    if price > current_price - ORDER_IMPROVE_MIN:
                        print(
                            f"[SKIP] update improve <{ORDER_IMPROVE_MIN:.2f}: "
                            f"{market_ticker} {side} current={current_price:.2f} target={price:.2f}"
                        )
                        continue
                    min_edge_dyn = _min_edge_for_prob(true_prob)
                    if (true_prob - price) < (min_edge_dyn + REPLACE_EDGE_BUFFER):
                        print(
                            f"[SKIP] replace edge<{(min_edge_dyn + REPLACE_EDGE_BUFFER):.3f}: "
                            f"{market_ticker} {side}"
                        )
                        continue

                    drift_key = f"drift:{market_ticker}:{side}"
                    drift_state = resting_state.get(drift_key, {})
                    drift_date = drift_state.get("date")
                    today = target_date_str
                    drift_total = drift_state.get("total", 0.0)
                    if drift_date != today:
                        drift_total = 0.0
                    drift_total += max(0.0, current_price - price)
                    if drift_total > MAX_REPLACE_DRIFT:
                        print(
                            f"[SKIP] replace drift>{MAX_REPLACE_DRIFT:.2f}: "
                            f"{market_ticker} {side}"
                        )
                        continue

                    if TRADE_MODE == "live":
                        try:
                            cancel_order(open_orders[(market_ticker, side)]["order_id"])
                            canceled_ids.add(
                                str(open_orders[(market_ticker, side)]["order_id"])
                            )
                            payload = {
                                "ticker": market_ticker,
                                "action": "buy",
                                "side": side.lower(),
                                "type": "limit",
                                "count": contracts,
                                "client_order_id": str(uuid.uuid4()),
                                "post_only": POST_ONLY,
                            }
                            if side == "YES":
                                payload["yes_price_dollars"] = _price_to_dollars(price)
                            else:
                                payload["no_price_dollars"] = _price_to_dollars(price)
                            resp = place_order(payload)
                            order_id = _extract_order_id(resp)
                            if order_id:
                                ledger[str(order_id)] = {"payload": payload, "response": resp}
                                save_ledger(ledger)
                            if order_id and row.get("book_odds") is not None:
                                resting_state.setdefault("order_book_odds", {})[
                                    str(order_id)
                                ] = row.get("book_odds")
                            resting_state[rest_key] = {
                                "last_update_ts": time.time(),
                                "updates": updates + 1,
                                "last_price": price,
                            }
                            resting_state[drift_key] = {
                                "date": today,
                                "total": drift_total,
                            }
                            save_resting_state(resting_state)
                            edge_after = calculate_edge(true_prob, price, KALSHI_FEE) * 100
                            _log_order_entry(
                                order_id,
                                market_ticker,
                                side,
                                contracts,
                                price,
                                edge_after / 100.0,
                                true_prob,
                                row.get("book_odds"),
                                row.get("market_type"),
                                row.get("total_line"),
                                row.get("team"),
                                row.get("event_ticker"),
                            )
                            print(
                                f"[INFO] Replaced order: {market_ticker} {side} "
                                f"{current_price:.2f} -> {price:.2f} x{contracts} "
                                f"| edge {edge_after:.2f}%"
                            )
                        except Exception as exc:
                            if "404" in str(exc):
                                print(f"[INFO] Replace skipped (order missing): {market_ticker} {side}")
                            else:
                                print(f"[WARN] Replace failed: {market_ticker} {side} -> {exc}")
                    continue

                # Position-aware add / new rules
                existing_pos = positions_by_market.get(market_ticker)
                event_key = row.get("event_ticker") or market_ticker
                event_dollars = positions_by_event.get(event_key, 0.0)
                event_dollars += open_order_dollars_by_event.get(event_key, 0.0)
                if event_dollars >= MAX_DOLLARS_PER_EVENT:
                    print(f"[SKIP] event cap ${MAX_DOLLARS_PER_EVENT:.2f}: {event_key}")
                    continue

                if existing_pos and existing_pos.get("count", 0) > 0:
                    if not ALLOW_POSITION_ADDS:
                        print(f"[SKIP] add disabled: {market_ticker} {side}")
                        continue
                    add_key = f"add:{market_ticker}:{side}:{target_date_str}"
                    adds = resting_state.get(add_key, 0)
                    if adds >= MAX_ADDS_PER_MARKET_PER_DAY:
                        print(f"[SKIP] max adds reached: {market_ticker} {side}")
                        continue
                    avg_price = existing_pos.get("avg_price") or price
                    # For both YES/NO, better means lower price
                    if (avg_price - price) < MIN_PRICE_IMPROVEMENT_ADD:
                        print(f"[SKIP] add not better by {MIN_PRICE_IMPROVEMENT_ADD:.2f}: {market_ticker}")
                        continue
                    if (true_prob - price) < MIN_EDGE_ADD:
                        print(f"[SKIP] add edge<{MIN_EDGE_ADD:.3f}: {market_ticker}")
                        continue
                else:
                    if (true_prob - price) < MIN_EDGE_NEW:
                        print(f"[SKIP] new edge<{MIN_EDGE_NEW:.3f}: {market_ticker}")
                        continue

                market_dollars = 0.0
                if existing_pos and existing_pos.get("count", 0) > 0:
                    market_dollars = (existing_pos.get("avg_price") or 0) * existing_pos.get("count", 0)
                market_dollars += open_order_dollars_by_market.get(market_ticker, 0.0)
                if (market_dollars + (price * contracts)) > MAX_DOLLARS_PER_MARKET:
                    print(f"[SKIP] market cap ${MAX_DOLLARS_PER_MARKET:.2f}: {market_ticker}")
                    continue

                payload = {
                    "ticker": market_ticker,
                    "action": "buy",
                    "side": side.lower(),
                    "type": "limit",
                    "count": contracts,
                    "client_order_id": str(uuid.uuid4()),
                    "post_only": POST_ONLY,
                }
                if side == "YES":
                    payload["yes_price_dollars"] = _price_to_dollars(price)
                else:
                    payload["no_price_dollars"] = _price_to_dollars(price)

                if TRADE_MODE == "live":
                    try:
                        resp = place_order(payload)
                        order_id = _extract_order_id(resp)
                        if order_id:
                            ledger[str(order_id)] = {
                                "payload": payload,
                                "response": resp,
                            }
                            save_ledger(ledger)
                        if order_id and row.get("book_odds") is not None:
                            resting_state.setdefault("order_book_odds", {})[
                                str(order_id)
                            ] = row.get("book_odds")
                            save_resting_state(resting_state)
                        if existing_pos and existing_pos.get("count", 0) > 0:
                            add_key = f"add:{market_ticker}:{side}:{target_date_str}"
                            resting_state[add_key] = resting_state.get(add_key, 0) + 1
                            save_resting_state(resting_state)
                        edge_after = calculate_edge(true_prob, price, KALSHI_FEE) * 100
                        _log_order_entry(
                            order_id,
                            market_ticker,
                            side,
                            contracts,
                            price,
                            edge_after / 100.0,
                            true_prob,
                            row.get("book_odds"),
                            row.get("market_type"),
                            row.get("total_line"),
                            row.get("team"),
                            row.get("event_ticker"),
                        )
                        exp_profit = row.get("expected_profit")
                        exp_profit_str = f"${exp_profit:.2f}" if exp_profit is not None else "n/a"
                        price_info = ""
                        if row.get("market_type") == "MONEYLINE" and side == "NO":
                            event_key = row.get("event_ticker") or market_ticker
                            team_norm = normalize_team(row.get("team"))
                            other_team = None
                            other_price = None
                            if event_key in event_team_yes:
                                for t_name, t_price in event_team_yes[event_key].items():
                                    if t_name != team_norm:
                                        other_team = t_name
                                        other_price = t_price
                                        break
                            if other_team and other_price is not None:
                                price_info = (
                                    f" | prices: {side} {team_norm}={price:.2f}, "
                                    f"YES {other_team}={other_price:.2f}"
                                )
                        book_odds_val = row.get("book_odds")
                        book_odds = _format_odds(book_odds_val)
                        p_true_str = f"{true_prob:.3f}" if true_prob is not None else "n/a"
                        p_book = None
                        try:
                            p_book = american_to_prob(float(book_odds_val))
                            p_book_str = f"{p_book:.3f}"
                        except Exception:
                            p_book_str = "n/a"
                        gap_str = (
                            f" | gap={abs(true_prob - p_book):.3f}"
                            if true_prob is not None and p_book is not None
                            else ""
                        )
                        books_info = "" if QUIET_LOGS else _format_books_info(row)
                        math_info = "" if QUIET_LOGS else _format_math_info(row)
                        if QUIET_LOGS:
                            gap_str = ""
                        print(
                            f"[INFO] Placed order: {market_ticker:<30} {side:<3} x{contracts:<3} "
                            f"| price {price:>5.2f} | edge {edge_after:>6.2f}% | exp {exp_profit_str:>8} | book {book_odds:>6}"
                            f" | p_true={p_true_str} | p_book={p_book_str}{gap_str}{math_info}{books_info}{price_info}"
                        )
                    except Exception as exc:
                        print(f"[WARN] Order failed: {market_ticker} {side} x{contracts} -> {exc}")
                        continue
                else:
                    price_info = ""
                    if row.get("market_type") == "MONEYLINE" and side == "NO":
                        event_key = row.get("event_ticker") or market_ticker
                        team_norm = normalize_team(row.get("team"))
                        other_team = None
                        other_price = None
                        if event_key in event_team_yes:
                            for t_name, t_price in event_team_yes[event_key].items():
                                if t_name != team_norm:
                                    other_team = t_name
                                    other_price = t_price
                                    break
                        if other_team and other_price is not None:
                            price_info = (
                                f" | prices: {side} {team_norm}={price:.2f}, "
                                f"YES {other_team}={other_price:.2f}"
                            )
                    edge_after = calculate_edge(true_prob, price, KALSHI_FEE) * 100
                    exp_profit = row.get("expected_profit")
                    exp_profit_str = f"${exp_profit:.2f}" if exp_profit is not None else "n/a"
                    book_odds_val = row.get("book_odds")
                    book_odds = _format_odds(book_odds_val)
                    p_true_str = f"{true_prob:.3f}" if true_prob is not None else "n/a"
                    p_book = None
                    try:
                        p_book = american_to_prob(float(book_odds_val))
                        p_book_str = f"{p_book:.3f}"
                    except Exception:
                        p_book_str = "n/a"
                    gap_str = (
                        f" | gap={abs(true_prob - p_book):.3f}"
                        if true_prob is not None and p_book is not None
                        else ""
                    )
                    books_info = "" if QUIET_LOGS else _format_books_info(row)
                    math_info = "" if QUIET_LOGS else _format_math_info(row)
                    if QUIET_LOGS:
                        gap_str = ""
                    print(
                        f"[INFO] DRY RUN order: {market_ticker:<30} {side:<3} x{contracts:<3} "
                        f"| price {price:>5.2f} | edge {edge_after:>6.2f}% | exp {exp_profit_str:>8} | book {book_odds:>6}"
                        f" | p_true={p_true_str} | p_book={p_book_str}{gap_str}{math_info}{books_info}{price_info}"
                    )
                orders_submitted += 1
                time.sleep(ORDER_SLEEP_SEC)

        if TRADE_MODE == "live":
            if canceled_ids:
                for oid in list(ledger.keys()):
                    if oid in canceled_ids:
                        ledger.pop(oid, None)
            save_ledger(ledger)

        if QUIET_LOGS:
            print(
                f"[INFO] Series counts: NCAAM markets={ncaam_markets}, NCAAW markets={ncaaw_markets} "
                f"| odds games: M={odds_m}, W={odds_w} | edges: M={edges_m}, W={edges_w}"
            )
            print(_edge_summary_line())
            print(_skip_summary_line())
            counts = _skip_summary_counts()
            total_blocked = sum(counts.values())
            if total_blocked == 0:
                print(f"[INFO] Placed={orders_submitted} | blocked: none")
            else:
                blocked_parts = ", ".join(
                    f"{k}={v}" for k, v in counts.items() if v > 0
                )
                print(f"[INFO] Placed={orders_submitted} | blocked: {blocked_parts}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    print(f"[INFO] Booting {Path(__file__).resolve()} | build=series-counts-2026-02-07")
    run()
