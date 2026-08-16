from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.sources import download, inspect_trade_gzip, orderbook_url, probe, trade_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2024-01-01")
    parser.add_argument("--output", default="outputs/reports/source_probe.json")
    parser.add_argument("--download-samples", action="store_true")
    parser.add_argument("--sample-bytes", type=int, default=50_000)
    args = parser.parse_args()

    trade = trade_url(args.date)
    orderbook = orderbook_url(args.date, depth=500)
    results = [probe(trade).__dict__, probe(orderbook).__dict__]
    payload: dict[str, object] = {"date": args.date, "probes": results}

    if args.download_samples:
        raw_dir = ROOT / "data" / "raw"
        trade_path = download(trade, raw_dir / f"BTCUSDT{args.date}.csv.gz", max_bytes=args.sample_bytes)
        orderbook_path = download(orderbook, raw_dir / f"{args.date}_BTCUSDT_ob500.data.zip", max_bytes=args.sample_bytes)
        payload["samples"] = {
            "trade_path": str(trade_path.relative_to(ROOT)),
            "orderbook_path": str(orderbook_path.relative_to(ROOT)),
            "trade_head": inspect_trade_gzip(trade_path, rows=5),
        }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

