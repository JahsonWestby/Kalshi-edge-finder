from pathlib import Path
import os


def _load_dotenv(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                keys.add(key)
                os.environ[key] = value
    except Exception:
        # best-effort only; do not block settings load
        return keys
    return keys


BASE_DIR = Path(__file__).resolve().parent.parent
_dotenv_path = BASE_DIR / ".env"
_dotenv_keys = _load_dotenv(_dotenv_path)
if _dotenv_path.exists():
    for _k in ("ODDS_API_KEY", "KALSHI_KEY_ID"):
        if _k not in _dotenv_keys:
            os.environ.pop(_k, None)

# bankroll
BANKROLL = 190.0
MAX_TRADE_SIZE = 15.0

# market toggles
ENABLE_MONEYLINE = True
ENABLE_TOTALS = True
MONEYLINE_SIDE_FILTER = "YES"  # "YES", "NO", or "ALL"
ENABLE_ARBS = True
USE_CHEAPEST_IMPLIED_WINNER = True

# core edge thresholds
MIN_EDGE = 0.015
MIN_EDGE_NBA = 0.025
MIN_EDGE_MLB = 0.015
TOTALS_MIN_EDGE = 0.04
AGGRESSIVE_EDGE = 0.05  # 5%
AGGRESSIVE_TICK = 0.01
EDGE_FAV_MIN = 0.015
EDGE_MID_MIN = 0.02
EDGE_DOG_MIN = 0.025
MIN_EDGE_NEW = 0.02
MIN_EDGE_ADD = 0.025
MAX_PROB_GAP = 0.06

# pricing / odds
KALSHI_FEE = 0.00
KALSHI_PRICE_MODE = "bid"  # "trade", "ask", "mid", or "bid"
YES_NO_TIE_PREFERENCE = 0.005
ASSUME_LIMIT_ORDER = True
POST_ONLY = True
MAX_KALSHI_PRICE = 0.95
MIN_VOLUME = 50

# sizing
KELLY_FRAC = 0.2
KELLY_CAP = 0.25
MAX_TRADE_PCT = 0.10
MAX_DOLLARS_PER_MARKET = 7.0
MAX_DOLLARS_PER_EVENT = 7.0
MAX_DOLLARS_PER_EVENT_TOTALS = 5.0
MAX_EVENT_PCT = 0.04
MAX_EVENT_PCT_TOTALS = 0.03

# order placement / updates
TRADE_MODE = "live"  # "dry-run" or "live"
MAX_ORDERS_PER_RUN = 30
ORDER_SLEEP_SEC = 0.12
ORDER_UPDATE_COOLDOWN_SEC = 45
MAX_UPDATES_PER_MARKET = 5
ORDER_IMPROVE_MIN = 0.01
MIN_PRICE_IMPROVEMENT_ADD = 0.01
MAX_ADDS_PER_MARKET_PER_DAY = 1
ALLOW_POSITION_ADDS = True

# arb limits
ARB_MIN_PROFIT = 0.10
ARB_MAX_CONTRACTS = 10
ARB_MAX_ORDERS_PER_RUN = 4

# hedge policy
HEDGE_POLICY = "arb_only"  # options: "off", "arb_only", "allow"
HEDGE_MIN_LOCKED_PROFIT_DOLLARS = 1.80
HEDGE_MIN_LOCKED_PROFIT_PCT_BR = 0.01
HEDGE_ALLOW_EXPOSURE_REDUCE = False
HEDGE_MAX_OPPOSITE_CONTRACTS = 0
ALLOW_TRUE_HEDGE = True

# cancel / replace rules
CANCEL_EV_BUFFER = -0.03
CANCEL_TIME_SEC = 60
OFF_MARKET_CENTS = 0.03        # 3¢
OFF_MARKET_TIME_SEC = 60       # 60s
CANCEL_OFFMARKET = OFF_MARKET_CENTS
CANCEL_NOT_COMPETITIVE_GAP = 0.01   # 1¢ behind best bid
CANCEL_NOT_COMPETITIVE_TIME_SEC = 30
CANCEL_NOT_COMPETITIVE_EV = 1.0     # legacy; EV gate effectively disabled
NO_BID_CANCEL_TIME_SEC = 90
NO_BID_KEEP_EV = 0.04
REPLACE_EDGE_BUFFER = 0.005
MAX_REPLACE_DRIFT = 0.05
GAME_START_CANCEL_MIN = 5
OPEN_ORDERS_STATUS = "open,resting,active"
RESTING_CANCEL_GRACE_SEC = 0

# totals matching
TOTALS_LINE_TOLERANCE = 0.1

# timing
POLL_INTERVAL = 30
ODDS_CACHE_TTL_SEC = 400
ODDS_CACHE_LOG = False
QUIET_LOGS = True
DATE_WINDOW_DAYS = 2

# secrets
KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID")  # e.g. your Kalshi API key id
KALSHI_PEM_PATH = BASE_DIR / "secrets" / "private_key.pem"

# Odds API settings
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_REGIONS = "us,uk,eu,fr,se,au"
# Sport keys (Odds API): basketball_ncaab, basketball_wncaab, basketball_nba, baseball_mlb, tennis_atp_*
ODDS_SPORTS = [
    "basketball_ncaab",
    "basketball_wncaab",
    "basketball_nba",
    "baseball_mlb",
]
ODDS_BOOKMAKERS = "lowvig,pinnacle,fanduel,bookmaker,betonlineag"

# Kalshi series tickers
MLB_SERIES_TICKER = "KXMLBSTGAME"
KALSHI_SERIES_TICKERS = [
    "KXNCAAMBGAME",
    "KXNCAAWBGAME",
    "KXNBAGAME",
    MLB_SERIES_TICKER,
    "KXATPMATCH",
]
KALSHI_TOTALS_SERIES_TICKERS = ["KXNCAAMBTOTAL"]
