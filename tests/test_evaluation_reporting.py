from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from data.event_io import write_events_jsonl
from lob.constants import CAPACITY
from lob.events import BookEvent
from reporting.evaluation_io import read_evaluation_episodes, write_evaluation_episodes
from reporting.time_of_day import EvaluationEpisode


def test_evaluation_episode_csv_round_trip(tmp_path: Path) -> None:
    episodes = [
        EvaluationEpisode(1_704_067_200_000, 0, 1.5, 0.2),
        EvaluationEpisode(1_704_070_800_000, 1, -0.5, 0.4),
    ]
    path = tmp_path / "eval.csv"
    write_evaluation_episodes(path, episodes)

    assert read_evaluation_episodes(path) == episodes


def test_time_of_day_script_writes_24_bucket_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    eval_path = tmp_path / "eval.csv"
    report_path = tmp_path / "tod.json"
    write_evaluation_episodes(
        eval_path,
        [
            EvaluationEpisode(1_704_067_200_000, 0, 1.0, 0.1),
            EvaluationEpisode(1_704_070_800_000, 1, 3.0, 0.3),
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "report_time_of_day.py"),
            "--evaluation-log",
            str(eval_path),
            "--report-output",
            str(report_path),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert len(report["buckets"]) == 24
    assert report["episode_count"] == 2


def test_tier1_replay_report_surfaces_capacity_overflow(tmp_path: Path) -> None:
    import shutil
    import tempfile

    root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "tier1_report.json"

    events = [BookEvent("ADD", "bid", 100.0, 1.0, order_id + 1, 0, 0) for order_id in range(CAPACITY)]
    events.append(BookEvent("ADD", "ask", 101.0, 1.0, 2001, 0, 0))
    events.append(BookEvent("ADD", "bid", 98.0, 1.0, 3001, 0, 1))

    fixture_dir = Path(tempfile.mkdtemp(prefix="tier1_overflow_test_", dir=str(root / "outputs")))
    try:
        events_path = fixture_dir / "overflow_events.jsonl"
        write_events_jsonl(events, events_path)

        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_tier1_replay.py"),
                "--events",
                str(events_path),
                "--report-output",
                str(report_path),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["passed"] is True
        assert report["capacity_overflow_occurred"] is True
        assert report["capacity_overflows"]["total"] >= 1
        assert report["capacity_overflows"]["bid"] >= 1
    finally:
        shutil.rmtree(fixture_dir, ignore_errors=True)
