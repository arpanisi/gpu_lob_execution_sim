from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_reconstructed_range_reports_missing_inputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_reconstructed_range.py"),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--events-dir",
            str(tmp_path / "events"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--manifest-output",
            str(manifest),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["daily_reports"][0]["date"] == "2024-01-01"
    assert payload["daily_reports"][0]["missing"]
