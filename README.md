**Overview**
Kalshi edge finder / market-making bot that compares Kalshi prices to book probabilities from the Odds API, places limit orders, manages resting orders, and optionally scans in-market arbs.

**Quick Start**
1. Create a `.env` file with your API keys:
```ini
KALSHI_KEY_ID=your_kalshi_key_id
ODDS_API_KEY=your_odds_api_key
```
2. Put your Kalshi private key here: `secrets/private_key.pem`
3. Run the bot:
```bash
python3.11 bot.py
```

**Configuration**
- `config/settings.py` — thresholds, polling, modes, sizing, and cancel/replace rules.
- `config/markets.yaml` — market/series filters.

**Project Layout**
- `bot.py` — main entrypoint
- `APIs/` — API clients (`kalshi_api.py`, `odds_api.py`)
- `logic/` — matchup/edge logic and helpers
- `strategy/` — sizing and strategy rules
- `scripts/` — one-off utilities and debug tools
- `data/` — caches, logs, and snapshots
- `secrets/` — private keys (ignored by git)
- `alerts.py` — alert hooks (optional)

**Data Outputs**
- `data/odds_cache.json` — cached Odds API responses
- `data/kalshi_cache.json` — cached Kalshi data
- `data/edges_found.csv` — recent edge findings
- `data/orders_log.csv` — order activity log
- `data/positions_snapshot.json` — snapshot of current positions
- `data/resting_orders.json` — cached resting orders
- `data/resting_orders_snapshot.json` — snapshot for review/debug
- `data/bets_ledger.json` — local order ledger
- `data/unmatched_teams.csv` / `data/unmatched_totals.csv` — matching diagnostics
- `data/*_debug.csv` — debug outputs (safe to delete)

If you delete files in `data/`, the bot will regenerate most of them on the next run, but you will lose local history/state for those files.

**Scripts**
- `scripts/kalshi_cancel_all_orders.py` — cancel all open orders
- `scripts/debug_unmatched.py` — dump unmatched teams/totals
- `scripts/odds_api_list_bookmakers.py` — list available bookmakers
- `scripts/odds_api_list_sports.py` — list available sports
- `scripts/odds_api_market_keys.py` — list market keys
- `scripts/odds_api_test_lowvig.py` — quick Odds API sanity check
- `scripts/kalshi_debug_endpoints.py` / `scripts/kalshi_debug_series_markets.py` — Kalshi debugging helpers
