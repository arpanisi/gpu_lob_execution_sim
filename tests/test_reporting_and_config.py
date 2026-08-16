from __future__ import annotations

from agents.ippo_config import LOCKED_IPPO_CONFIG, convergence_triggered
from reporting.time_of_day import EvaluationEpisode, hour_of_day_breakdown


def test_locked_ippo_config_matches_plan_values() -> None:
    config = LOCKED_IPPO_CONFIG
    assert config.agents_per_environment == 2
    assert config.learning_rate == 3e-4
    assert config.rollout_length == 128
    assert config.minibatch_size == 256
    assert config.update_epochs == 10
    assert config.clip_ratio == 0.2
    assert config.gae_lambda == 0.95
    assert config.discount == 0.999
    assert config.entropy_coefficient == 0.001
    assert config.eval_interval_steps == 50_000
    assert config.heldout_eval_episodes == 200
    assert config.max_environment_steps == 5_000_000


def test_convergence_rule_uses_100_checkpoint_average_and_20_checkpoint_lookback() -> None:
    rewards = [100.0] * 100 + [100.5] * 20
    assert convergence_triggered(rewards) is True

    improved = [100.0] * 100 + [110.0] * 20
    assert convergence_triggered(improved) is False


def test_hour_of_day_report_always_returns_24_buckets() -> None:
    episodes = [
        EvaluationEpisode(1_704_067_200_000, 0, 1.0, 0.1),
        EvaluationEpisode(1_704_070_800_000, 0, 3.0, 0.3),
        EvaluationEpisode(1_704_070_900_000, 1, 5.0, 0.5),
    ]
    rows = hour_of_day_breakdown(episodes)

    assert len(rows) == 24
    assert rows[0]["episodes"] == 1
    assert rows[1]["episodes"] == 2
    assert rows[1]["mean_reward"] == 4.0
    assert rows[2]["mean_reward"] is None
