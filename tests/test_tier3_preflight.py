from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_preflight_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prepare_tier3_full_run", root / "scripts" / "prepare_tier3_full_run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_tier3_speedup_gate_requires_all_locked_batch_sizes() -> None:
    module = _load_preflight_module()
    assert module._speedup_ratio_available({"measurements": [{"batch_size": 16, "speedup_ratio": 1.1}]}) is False
    assert (
        module._speedup_ratio_available(
            {
                "measurements": [
                    {"batch_size": 16, "speedup_ratio": 1.1},
                    {"batch_size": 64, "speedup_ratio": 1.2},
                    {"batch_size": 256, "speedup_ratio": 1.3},
                ]
            }
        )
        is True
    )
