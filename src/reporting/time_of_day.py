from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class EvaluationEpisode:
    start_timestamp_ms: int
    agent_id: int
    episode_reward: float
    realized_shortfall: float


def hour_of_day_breakdown(episodes: list[EvaluationEpisode]) -> list[dict[str, float | int | None]]:
    grouped: dict[int, list[EvaluationEpisode]] = defaultdict(list)
    for episode in episodes:
        hour = datetime.fromtimestamp(episode.start_timestamp_ms / 1000.0, tz=timezone.utc).hour
        grouped[hour].append(episode)

    rows: list[dict[str, float | int | None]] = []
    for hour in range(24):
        bucket = grouped[hour]
        if not bucket:
            rows.append({"hour_utc": hour, "episodes": 0, "mean_reward": None, "mean_realized_shortfall": None})
            continue
        rows.append(
            {
                "hour_utc": hour,
                "episodes": len(bucket),
                "mean_reward": sum(item.episode_reward for item in bucket) / len(bucket),
                "mean_realized_shortfall": sum(item.realized_shortfall for item in bucket) / len(bucket),
            }
        )
    return rows
