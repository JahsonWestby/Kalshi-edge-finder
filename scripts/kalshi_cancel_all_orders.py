import argparse
import os
import sys

# Ensure repo root is on path when running from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from APIs.kalshi_api import get_orders, cancel_order


def _get_open_orders():
    orders = get_orders()
    items = orders.get("orders") or []
    open_orders = []
    for o in items:
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
        open_orders.append(o)
    return open_orders


def _filter_by_prefixes(open_orders, prefixes):
    return [
        o for o in open_orders
        if any(
            (o.get("market_ticker") or o.get("ticker") or "").startswith(p)
            for p in prefixes
        )
    ]


def _cancel_orders(matched):
    canceled = 0
    for o in matched:
        order_id = o.get("order_id") or o.get("id")
        if not order_id:
            continue
        try:
            cancel_order(order_id)
            canceled += 1
        except Exception as exc:
            print(f"[WARN] Failed to cancel {order_id}: {exc}")
    return canceled


def main():
    parser = argparse.ArgumentParser(description="Cancel Kalshi open orders by ticker prefix.")
    parser.add_argument(
        "--prefixes", nargs="+", default=None,
        help="Ticker prefixes to cancel (e.g. KXNBAGAME KXNCAAMBGAME). "
             "Defaults to KXNCAA for interactive mode.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompt (non-interactive / shutdown mode).",
    )
    args = parser.parse_args()

    open_orders = _get_open_orders()

    if args.yes:
        # Non-interactive mode — called by shutdown runner
        prefixes = args.prefixes or []
        if not prefixes:
            print("[WARN] --yes passed but no --prefixes given; nothing to cancel.")
            return
        matched = _filter_by_prefixes(open_orders, prefixes)
        canceled = _cancel_orders(matched)
        print(f"[INFO] Shutdown cancel: {canceled}/{len(matched)} orders canceled "
              f"(prefixes: {prefixes})")
        return

    # Interactive mode (manual use)
    prefixes = args.prefixes or ["KXNCAA"]
    matched = _filter_by_prefixes(open_orders, prefixes)
    print(f"Open orders total: {len(open_orders)} | matching {prefixes}: {len(matched)}")
    if not matched:
        return
    confirm = input(f"Type YES to cancel {len(matched)} matching orders: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return
    canceled = _cancel_orders(matched)
    print(f"Canceled {canceled}/{len(matched)} orders.")


if __name__ == "__main__":
    main()
