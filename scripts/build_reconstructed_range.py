from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.date_ranges import iter_dates
from data.event_io import write_events_jsonl
from data.orderbook import iter_bybit_orderbook_zip
from data.reconstruct import reconstruct_events
from data.sources import download, orderbook_url, trade_url
from data.trades import load_bybit_trade_gzip
from data.validation import validate_replay, validate_trade_consumption


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--events-dir", default="outputs/events")
    parser.add_argument("--reports-dir", default="outputs/reports/reconstruction")
    parser.add_argument("--manifest-output", default="outputs/reports/reconstruction_manifest.json")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--max-records-per-day", type=int, default=None)
    args = parser.parse_args()

    raw_dir = _project_path(args.raw_dir)
    events_dir = _project_path(args.events_dir)
    reports_dir = _project_path(args.reports_dir)
    dates = iter_dates(args.start_date, args.end_date)
    daily_reports = []
    for day in dates:
        orderbook_path = raw_dir / f"{day}_BTCUSDT_ob500.data.zip"
        trade_path = raw_dir / f"BTCUSDT{day}.csv.gz"
        if args.download_missing:
            if not orderbook_path.exists():
                download(orderbook_url(day), orderbook_path)
            if not trade_path.exists():
                download(trade_url(day), trade_path)
        missing = [_display_path(path) for path in (orderbook_path, trade_path) if not path.exists()]
        if missing:
            daily_reports.append({"date": day, "passed": False, "missing": missing})
            continue
        daily_reports.append(_reconstruct_one_day(day, orderbook_path, trade_path, events_dir, reports_dir, args.max_records_per_day))

    manifest = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "dates": dates,
        "download_missing": args.download_missing,
        "max_records_per_day": args.max_records_per_day,
        "daily_reports": daily_reports,
        "passed": all(item.get("passed") for item in daily_reports),
    }
    output = _project_path(args.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


def _reconstruct_one_day(
    day: str,
    orderbook_path: Path,
    trade_path: Path,
    events_dir: Path,
    reports_dir: Path,
    max_records: int | None,
) -> dict:
    depth_records = list(iter_bybit_orderbook_zip(orderbook_path, max_records=max_records))
    if not depth_records:
        return {"date": day, "passed": False, "error": "no order-book records parsed"}
    start_ts = depth_records[0].timestamp
    end_ts = depth_records[-1].timestamp
    trades = [trade for trade in load_bybit_trade_gzip(trade_path) if start_ts - 1_000 <= trade.timestamp <= end_ts + 1_000]
    result = reconstruct_events(depth_records, trades)
    events_path = events_dir / f"reconstructed_{day}.jsonl"
    report_path = reports_dir / f"reconstruction_{day}.json"
    write_events_jsonl(result.events, events_path)
    replay = validate_replay(result.events, depth_records)
    consumption = validate_trade_consumption({trade.trade_id: trade.quantity for trade in trades}, result.trade_consumed)
    report = {
        "date": day,
        "orderbook_zip": _display_path(orderbook_path),
        "trades_gzip": _display_path(trade_path),
        "max_records": max_records,
        "depth_records": len(depth_records),
        "trades_in_window": len(trades),
        "event_count": len(result.events),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "events_output": _display_path(events_path),
        "report_output": _display_path(report_path),
        "replay_validation": replay.__dict__,
        "trade_consumption_validation": consumption.__dict__,
        "passed": replay.passed and consumption.passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
