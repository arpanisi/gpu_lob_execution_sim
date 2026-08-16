from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reporting.evaluation_io import read_evaluation_episodes
from reporting.time_of_day import hour_of_day_breakdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-log", default="outputs/evaluation/tier3_eval_episodes.csv")
    parser.add_argument("--report-output", default="outputs/reports/time_of_day_breakdown.json")
    args = parser.parse_args()

    evaluation_log = _project_path(args.evaluation_log)
    episodes = read_evaluation_episodes(evaluation_log)
    rows = hour_of_day_breakdown(episodes)
    report = {
        "evaluation_log": _display_path(evaluation_log),
        "episode_count": len(episodes),
        "buckets": rows,
        "passed": len(rows) == 24,
    }
    output = _project_path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


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
