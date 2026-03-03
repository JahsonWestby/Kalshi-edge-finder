import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

INTERVAL_SEC = 60 * 60  # 1 hour
COOLDOWN_SEC = 5
CRASH_THRESHOLD_SEC = 300  # if bot exits in < 5 min, treat as crash
CRASH_RETRY_SEC = 60       # wait 60s before retrying after a crash

# Only run the bot during this local hour window (24h clock, inclusive start, exclusive end)
ACTIVE_START_HOUR = 9   # 9am
ACTIVE_END_HOUR = 25    # 1am next day (9 + 16 = 25)

# Cancel all open orders for tracked sports on window exit
CANCEL_ON_SHUTDOWN = True
CANCEL_PREFIXES = [
    "KXNCAAMBGAME",   # NCAAB moneylines
    "KXNCAAWBGAME",   # NCAAW moneylines
    "KXNBAGAME",      # NBA moneylines
    "KXMLBSTGAME",    # MLB moneylines
    "KXATPMATCH",     # ATP tennis
    "KXWTAMATCH",     # WTA tennis
    "KXNCAAMBTOTAL",  # NCAAB totals
]


def _bot_path() -> Path:
    return Path(__file__).resolve().parents[1] / "bot.py"


def _cancel_script_path() -> Path:
    return Path(__file__).resolve().parent / "kalshi_cancel_all_orders.py"


def _run_shutdown_cancel(python_bin: str) -> None:
    print("[INFO] Shutdown: canceling open orders for tracked sports...")
    cmd = [python_bin, str(_cancel_script_path()), "--yes", "--prefixes"] + CANCEL_PREFIXES
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stdout.splitlines():
            print(line)
        if result.returncode != 0 and result.stderr:
            print(f"[WARN] Cancel script stderr: {result.stderr.strip()}")
    except Exception as exc:
        print(f"[WARN] Shutdown cancel failed: {exc}")


def main() -> None:
    bot_path = _bot_path()
    python_bin = sys.executable or "python3"
    print(f"[INFO] Hourly runner using: {python_bin}")
    print(f"[INFO] Bot: {bot_path}")

    was_in_window = False

    while True:
        hour = datetime.now().hour
        # Treat hours past midnight (0-8) as 24-32 for easy comparison
        adjusted = hour if hour >= ACTIVE_START_HOUR else hour + 24
        in_window = ACTIVE_START_HOUR <= adjusted < ACTIVE_END_HOUR

        if not in_window:
            if was_in_window and CANCEL_ON_SHUTDOWN:
                _run_shutdown_cancel(python_bin)
            was_in_window = False
            print(f"[INFO] Outside active window ({ACTIVE_START_HOUR}:00–{ACTIVE_END_HOUR % 24:02d}:00), sleeping 5min.")
            time.sleep(300)
            continue

        was_in_window = True
        start = time.time()
        try:
            print("[INFO] Starting bot...")
            subprocess.run([python_bin, str(bot_path)], timeout=INTERVAL_SEC)
            print("[INFO] Bot exited.")
        except subprocess.TimeoutExpired:
            print("[INFO] Bot reached hourly limit, restarting.")
        except Exception as exc:
            print(f"[WARN] Bot runner error: {exc}")

        elapsed = time.time() - start
        if elapsed < CRASH_THRESHOLD_SEC:
            print(f"[WARN] Bot exited after {int(elapsed)}s (possible crash), retrying in {CRASH_RETRY_SEC}s.")
            time.sleep(CRASH_RETRY_SEC)
        else:
            sleep_for = max(0, INTERVAL_SEC - elapsed)
            if sleep_for > 0:
                print(f"[INFO] Sleeping {int(sleep_for)}s before restart.")
                time.sleep(sleep_for)
        time.sleep(COOLDOWN_SEC)


if __name__ == "__main__":
    main()
