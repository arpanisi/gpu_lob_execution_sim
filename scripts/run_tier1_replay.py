from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.event_io import read_events_jsonl
from env.episode import observations, reset, step
from env.execution import AgentAction
from env.windows import slice_non_overlapping_windows
from lob.constants import EPISODE_LENGTH_MS, STEP_EVENT_COUNT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="outputs/events/reconstructed_2024-01-01_head.jsonl")
    parser.add_argument("--report-output", default="outputs/reports/tier1_replay_report.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--episode-length-ms", type=int, default=EPISODE_LENGTH_MS)
    parser.add_argument("--k-events", type=int, default=STEP_EVENT_COUNT)
    args = parser.parse_args()

    events_path = _project_path(args.events)
    report_path = _project_path(args.report_output)
    events = read_events_jsonl(events_path)
    windows = slice_non_overlapping_windows(events, episode_length_ms=args.episode_length_ms)
    if not windows:
        raise SystemExit(f"no usable episode windows in {events_path}")

    states = [reset(windows[idx % len(windows)], agent_sides=("buy",), target_quantity=0.05) for idx in range(args.batch_size)]
    total_steps = 0
    total_fills = 0
    total_reward = 0.0
    completed = 0
    observation_lengths: set[int] = set()
    for _ in range(args.max_steps):
        active = False
        for state in states:
            if state.done:
                continue
            active = True
            obs = observations(state)
            observation_lengths.update(len(item) for item in obs)
            result = step(state, [AgentAction(offset_ticks=0, quantity=0.01)], k_events=args.k_events)
            total_steps += 1
            total_fills += len(result.fills)
            total_reward += result.rewards[0]
            if result.done:
                completed += 1
        if not active:
            break

    total_bid_overflows = sum(state.book.bid_overflows for state in states)
    total_ask_overflows = sum(state.book.ask_overflows for state in states)
    capacity_overflow_occurred = (total_bid_overflows + total_ask_overflows) > 0

    report = {
        "events": str(events_path.relative_to(ROOT)),
        "event_count": len(events),
        "windows": len(windows),
        "batch_size": args.batch_size,
        "steps_executed": total_steps,
        "completed_episodes": completed,
        "fills": total_fills,
        "mean_reward_per_step": total_reward / total_steps if total_steps else 0.0,
        "observation_lengths": sorted(observation_lengths),
        "capacity_overflows": {
            "bid": total_bid_overflows,
            "ask": total_ask_overflows,
            "total": total_bid_overflows + total_ask_overflows,
        },
        "capacity_overflow_occurred": capacity_overflow_occurred,
        "passed": total_steps > 0 and observation_lengths == {89},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if capacity_overflow_occurred:
        print(
            "WARNING: capacity overflow occurred during Tier 1 replay "
            f"({total_bid_overflows + total_ask_overflows} dropped orders, "
            f"{total_bid_overflows} bid / {total_ask_overflows} ask).",
            file=sys.stderr,
        )
    if not report["passed"]:
        raise SystemExit(1)


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
