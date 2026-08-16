from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.event_io import write_events_jsonl
from data.orderbook import iter_bybit_orderbook_zip
from data.reconstruct import reconstruct_events
from data.trades import load_bybit_trade_gzip
from data.validation import validate_replay, validate_trade_consumption


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orderbook-zip", required=True)
    parser.add_argument("--trades-gzip", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--events-output", default="outputs/events/reconstructed_events.jsonl")
    parser.add_argument("--report-output", default="outputs/reports/reconstruction_report.json")
    args = parser.parse_args()

    depth_records = list(iter_bybit_orderbook_zip(args.orderbook_zip, max_records=args.max_records))
    if not depth_records:
        raise SystemExit("no order-book records parsed")
    start_ts = depth_records[0].timestamp
    end_ts = depth_records[-1].timestamp
    trades = [trade for trade in load_bybit_trade_gzip(args.trades_gzip) if start_ts - 1_000 <= trade.timestamp <= end_ts + 1_000]
    result = reconstruct_events(depth_records, trades)

    events_path = _project_path(args.events_output)
    report_path = _project_path(args.report_output)
    write_events_jsonl(result.events, events_path)
    replay = validate_replay(result.events, depth_records)
    consumption = validate_trade_consumption({trade.trade_id: trade.quantity for trade in trades}, result.trade_consumed)
    report = {
        "orderbook_zip": args.orderbook_zip,
        "trades_gzip": args.trades_gzip,
        "max_records": args.max_records,
        "depth_records": len(depth_records),
        "trades_in_window": len(trades),
        "event_count": len(result.events),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "events_output": str(events_path.relative_to(ROOT)),
        "replay_validation": replay.__dict__,
        "trade_consumption_validation": consumption.__dict__,
        "passed": replay.passed and consumption.passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def _project_path(path: str) -> Path:
    output = Path(path)
    return output if output.is_absolute() else ROOT / output


if __name__ == "__main__":
    main()

