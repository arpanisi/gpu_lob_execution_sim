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
    parser.add_argument("--report-output", default="outputs/reports/tier2_ippo_preflight.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--pattern", default="reconstructed_2024-01-*.jsonl")
    args = parser.parse_args()

    events_dir = _project_path(args.events_dir)
    files = sorted(events_dir.glob(args.pattern))
    config = LOCKED_IPPO_CONFIG
    report = {
        "events_dir": str(events_dir.relative_to(ROOT)),
        "pattern": args.pattern,
        "event_files": [str(path.relative_to(ROOT)) for path in files],
        "batch_size": args.batch_size,
        "agents": args.agents,
        "ippo_config": config.__dict__,
        "ready_for_short_training": bool(files) and args.batch_size == 64 and args.agents == config.agents_per_environment,
        "training_launched": False,
    }

    output = _project_path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_short_training"]:
        raise SystemExit(1)


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
