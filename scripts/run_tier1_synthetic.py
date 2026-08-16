from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.event_io import write_events_jsonl
from data.reconstruct import reconstruct_events
from data.validation import validate_replay, validate_trade_consumption
from lob.events import DepthRecord, TradePrint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-output", default="outputs/events/tier1_synthetic_events.jsonl")
    parser.add_argument("--report-output", default="outputs/reports/tier1_synthetic_validation.json")
    args = parser.parse_args()

    depth, trades = _synthetic_depth_and_trades()
    result = reconstruct_events(depth, trades)
    events_path = _project_path(args.events_output)
    report_path = _project_path(args.report_output)
    write_events_jsonl(result.events, events_path)
    replay = validate_replay(result.events, depth)
    trade_quantities = {trade.trade_id: trade.quantity for trade in trades}
    consumption = validate_trade_consumption(trade_quantities, result.trade_consumed)
    report = {
        "events_output": str(events_path.relative_to(ROOT)),
        "event_count": len(result.events),
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


def _synthetic_depth_and_trades() -> tuple[list[DepthRecord], list[TradePrint]]:
    depth = [
        DepthRecord(1_704_067_200_000, bids=((42_324.9, 2.0),), asks=((42_325.0, 1.0),), is_snapshot=True),
        DepthRecord(1_704_067_200_200, bids=((42_324.9, 0.5),), asks=((42_325.0, 1.25),)),
        DepthRecord(1_704_067_200_400, bids=((42_324.9, 0.0),), asks=((42_325.0, 1.25),)),
    ]
    trades = [
        TradePrint(1_704_067_200_235, price=42_324.9, quantity=1.0, aggressor_side="sell", trade_id="synthetic_t1"),
    ]
    return depth, trades


if __name__ == "__main__":
    main()

