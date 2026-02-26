import os
import sys
import json

# Ensure repo root is on path when running from scripts/NCA
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from APIs.kalshi_api import get_orders, cancel_order


def main():
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

    ncaa_prefixes = ("KXNCAA",)
    ncaa_orders = []
    for o in open_orders:
        ticker = (o.get("market_ticker") or o.get("ticker") or "")
        if ticker.startswith(ncaa_prefixes):
            ncaa_orders.append(o)

    print(
        json.dumps(
            {
                "open_orders": len(open_orders),
                "ncaa_orders": len(ncaa_orders),
                "ncaa_prefixes": list(ncaa_prefixes),
            },
            indent=2,
        )
    )
    if not open_orders:
        return

    confirm = input("Type NCAA to cancel all NCAA open orders: ").strip()
    if confirm != "NCAA":
        print("Aborted.")
        return

    canceled = 0
    for o in ncaa_orders:
        order_id = o.get("order_id") or o.get("id")
        if not order_id:
            continue
        try:
            cancel_order(order_id)
            canceled += 1
        except Exception as exc:
            print(f"Failed to cancel {order_id}: {exc}")

    print(f"Canceled {canceled} orders.")


if __name__ == "__main__":
    main()
