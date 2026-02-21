from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import csv
import difflib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from APIs.odds_api import get_moneyline_games, get_totals_games
from APIs.kalshi_api import get_kalshi_markets, get_kalshi_totals_markets
from logic.normalize import normalize_team, normalize_player, normalize_nba_team
from config.settings import DATE_WINDOW_DAYS

SERIES_BY_SPORT = {
    "basketball_ncaab": "KXNCAAMBGAME",
    "basketball_wncaab": "KXNCAAWBGAME",
    "basketball_nba": "KXNBAGAME",
}


def _series_for_sport(sport_key: str | None) -> str | None:
    if not sport_key:
        return None
    if sport_key.startswith("tennis_atp_"):
        return "KXATPMATCH"
    if sport_key.startswith("tennis_wta_"):
        return "KXWTAMATCH"
    return SERIES_BY_SPORT.get(sport_key)


def _normalize_for_series(series: str, name: str) -> str:
    if series in {"KXATPMATCH", "KXATPGAME", "KXWTAMATCH", "KXWTAGAME"}:
        return normalize_player(name)
    if series == "KXNBAGAME":
        return normalize_nba_team(name)
    return normalize_team(name)


def _matchup_key(away: str, home: str) -> str:
    return f"{away}@{home}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug unmatched Odds API vs Kalshi teams")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tomorrow", action="store_true", help="Use tomorrow as target date.")
    parser.add_argument("--window", type=int, default=DATE_WINDOW_DAYS, help="Date window days.")
    parser.add_argument("--out", default="data/unmatched_teams_debug.csv", help="CSV output path.")
    parser.add_argument(
        "--out-totals",
        default="data/unmatched_totals_debug.csv",
        help="Totals CSV output path.",
    )
    args = parser.parse_args()

    if args.tomorrow:
        target_date = datetime.now().astimezone().date() + timedelta(days=1)
    elif args.date:
        target_date = datetime.fromisoformat(args.date).date()
    else:
        target_date = datetime.now().astimezone().date()

    games = get_moneyline_games(
        target_date=target_date,
        tz_name="US/Central",
        date_window_days=args.window,
    )
    totals_games = get_totals_games(
        target_date=target_date,
        tz_name="US/Central",
        date_window_days=args.window,
    )

    odds_by_series: dict[str, set[str]] = {}
    for g in games:
        series = _series_for_sport(g.get("sport_key"))
        if not series:
            continue
        home = _normalize_for_series(series, g["home"])
        away = _normalize_for_series(series, g["away"])
        odds_by_series.setdefault(series, set()).update({home, away})

    kalshi_by_series = get_kalshi_markets(
        target_date=target_date,
        tz_name="US/Central",
        date_window_days=args.window,
    )
    kalshi_teams = {s: set(v.keys()) for s, v in kalshi_by_series.items()}

    rows = []
    kalshi_keys_all = {s: sorted(v.keys()) for s, v in kalshi_by_series.items()}
    for series, odds_teams in odds_by_series.items():
        missing = sorted(t for t in odds_teams if t not in kalshi_teams.get(series, set()))
        print(f"[INFO] {series}: odds={len(odds_teams)} kalshi={len(kalshi_teams.get(series, set()))} missing={len(missing)}")
        for t in missing:
            matches = difflib.get_close_matches(t, kalshi_keys_all.get(series, []), n=3, cutoff=0.6)
            while len(matches) < 3:
                matches.append("")
            rows.append(
                {
                    "series": series,
                    "odds_team": t,
                    "closest_kalshi_1": matches[0],
                    "closest_kalshi_2": matches[1],
                    "closest_kalshi_3": matches[2],
                }
            )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "odds_team",
                "closest_kalshi_1",
                "closest_kalshi_2",
                "closest_kalshi_3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote unmatched teams: {out_path}")

    # Totals unmatched by matchup key
    kalshi_totals = get_kalshi_totals_markets(
        target_date=target_date,
        tz_name="US/Central",
        date_window_days=args.window,
    )
    kalshi_totals_keys = set()
    kalshi_totals_lines: dict[str, list[float]] = {}
    for m in kalshi_totals:
        key = m.get("matchup_key")
        line = m.get("total")
        if key:
            kalshi_totals_keys.add(key)
            if line is not None:
                kalshi_totals_lines.setdefault(key, []).append(float(line))

    totals_rows = []
    for g in totals_games:
        key = _matchup_key(g["away"], g["home"])
        if key not in kalshi_totals_keys:
            totals_rows.append(
                {
                    "matchup_key": key,
                    "total": g.get("total"),
                    "closest_kalshi_line": "",
                    "line_diff": "",
                }
            )
            continue
        lines = kalshi_totals_lines.get(key) or []
        total_val = g.get("total")
        closest_line = ""
        line_diff = ""
        if lines and total_val is not None:
            try:
                total_float = float(total_val)
                closest = min(lines, key=lambda v: abs(v - total_float))
                closest_line = f"{closest:.1f}".rstrip("0").rstrip(".")
                line_diff = f"{abs(closest - total_float):.2f}".rstrip("0").rstrip(".")
            except Exception:
                pass
        totals_rows.append(
            {
                "matchup_key": key,
                "total": total_val,
                "closest_kalshi_line": closest_line,
                "line_diff": line_diff,
            }
        )

    out_totals = ROOT / args.out_totals
    out_totals.parent.mkdir(parents=True, exist_ok=True)
    with out_totals.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["matchup_key", "total", "closest_kalshi_line", "line_diff"]
        )
        writer.writeheader()
        writer.writerows(totals_rows)
    print(f"[INFO] Wrote unmatched totals: {out_totals}")


if __name__ == "__main__":
    main()
