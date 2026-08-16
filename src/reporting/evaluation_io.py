from __future__ import annotations

import csv
from pathlib import Path

from reporting.time_of_day import EvaluationEpisode


def read_evaluation_episodes(path: str | Path) -> list[EvaluationEpisode]:
    source = Path(path)
    episodes: list[EvaluationEpisode] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"start_timestamp_ms", "agent_id", "episode_reward", "realized_shortfall"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"evaluation log missing columns: {sorted(missing)}")
        for row in reader:
            episodes.append(
                EvaluationEpisode(
                    start_timestamp_ms=int(row["start_timestamp_ms"]),
                    agent_id=int(row["agent_id"]),
                    episode_reward=float(row["episode_reward"]),
                    realized_shortfall=float(row["realized_shortfall"]),
                )
            )
    return episodes


def write_evaluation_episodes(path: str | Path, episodes: list[EvaluationEpisode]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_timestamp_ms", "agent_id", "episode_reward", "realized_shortfall"])
        writer.writeheader()
        for episode in episodes:
            writer.writerow(
                {
                    "start_timestamp_ms": episode.start_timestamp_ms,
                    "agent_id": episode.agent_id,
                    "episode_reward": episode.episode_reward,
                    "realized_shortfall": episode.realized_shortfall,
                }
            )
    return output
