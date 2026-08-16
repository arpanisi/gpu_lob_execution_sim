from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.ippo_config import LOCKED_IPPO_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", default="outputs/events")
    parser.add_argument("--pattern", default="reconstructed_2024-*.jsonl")
    parser.add_argument("--tier1-report", default="outputs/reports/tier1_replay_report.json")
    parser.add_argument("--tier2-report", default="outputs/reports/tier2_ippo_report.json")
    parser.add_argument("--speedup-report", default="outputs/reports/step3_speedup.json")
    parser.add_argument("--report-output", default="outputs/reports/tier3_preflight.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--agents", type=int, default=2)
    args = parser.parse_args()

    event_files = sorted(_project_path(args.events_dir).glob(args.pattern))
    tier1 = _load_json(_project_path(args.tier1_report))
    tier2 = _load_json(_project_path(args.tier2_report))
    speedup = _load_json(_project_path(args.speedup_report))
    acceptance = {
        "tier1_passed": bool(tier1 and tier1.get("passed")),
        "tier2_acceptance_passed": bool(tier2 and all((tier2.get("acceptance") or {}).values())),
        "speedup_ratios_recorded": _speedup_ratio_available(speedup),
        "full_year_events_available": len(event_files) >= 365,
        "batch_size_locked": args.batch_size == 256,
        "agents_locked": args.agents == LOCKED_IPPO_CONFIG.agents_per_environment,
    }
    report = {
        "events_dir": str(_project_path(args.events_dir).relative_to(ROOT)),
        "pattern": args.pattern,
        "event_file_count": len(event_files),
        "sample_event_files": [str(path.relative_to(ROOT)) for path in event_files[:5]],
        "batch_size": args.batch_size,
        "agents": args.agents,
        "ippo_config": LOCKED_IPPO_CONFIG.__dict__,
        "acceptance": acceptance,
        "ready_for_tier3": all(acceptance.values()),
    }
    output = _project_path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_tier3"]:
        raise SystemExit(1)


def _speedup_ratio_available(speedup: dict | None) -> bool:
    if not speedup:
        return False
    measurements = speedup.get("measurements", [])
    required = {16, 64, 256}
    seen = {int(row.get("batch_size")) for row in measurements if row.get("speedup_ratio") is not None}
    return required.issubset(seen)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
