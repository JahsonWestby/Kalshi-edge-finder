import subprocess
import sys
import time
from pathlib import Path

INTERVAL_SEC = 60 * 60  # 1 hour
COOLDOWN_SEC = 5


def _bot_path() -> Path:
    return Path(__file__).resolve().parents[1] / "bot.py"


def main() -> None:
    bot_path = _bot_path()
    python_bin = sys.executable or "python3"
    print(f"[INFO] Hourly runner using: {python_bin}")
    print(f"[INFO] Bot: {bot_path}")

    while True:
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
        sleep_for = max(0, INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            print(f"[INFO] Sleeping {int(sleep_for)}s before restart.")
            time.sleep(sleep_for)
        time.sleep(COOLDOWN_SEC)


if __name__ == "__main__":
    main()
