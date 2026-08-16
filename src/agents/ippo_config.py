from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    agents_per_environment: int = 2
    learning_rate: float = 3e-4
    rollout_length: int = 128
    minibatch_size: int = 256
    update_epochs: int = 10
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    discount: float = 0.999
    entropy_coefficient: float = 0.001
    eval_interval_steps: int = 50_000
    heldout_eval_episodes: int = 200
    convergence_window: int = 100
    convergence_lookback: int = 20
    convergence_min_relative_improvement: float = 0.01
    max_environment_steps: int = 5_000_000


LOCKED_IPPO_CONFIG = PPOConfig()


def convergence_triggered(mean_rewards: list[float], config: PPOConfig = LOCKED_IPPO_CONFIG) -> bool:
    required = config.convergence_window + config.convergence_lookback
    if len(mean_rewards) < required:
        return False
    current = sum(mean_rewards[-config.convergence_window :]) / config.convergence_window
    previous_end = -config.convergence_lookback
    previous_start = previous_end - config.convergence_window
    previous = sum(mean_rewards[previous_start:previous_end]) / config.convergence_window
    if previous == 0:
        return abs(current) < config.convergence_min_relative_improvement
    relative_improvement = (current - previous) / abs(previous)
    return relative_improvement < config.convergence_min_relative_improvement
