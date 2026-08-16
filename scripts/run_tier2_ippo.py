from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.ippo_config import LOCKED_IPPO_CONFIG
from agents.jax_ippo import action_summary, init_adam, init_two_agent_params, ppo_update, require_ippo_jax, sample_action
from data.event_io import read_events_jsonl
from data.informed_flow import compute_informed_flow_signals, flow_value_series
from env.jax_batched_env import batched_observations, batched_step, current_batched_flow_values, make_batched_state
from env.windows import slice_non_overlapping_windows
from lob.constants import EPISODE_LENGTH_MS
from lob.jax_matching import batched_apply_event_arrays


def main() -> None:
    require_ippo_jax()
    import jax
    import jax.numpy as jnp

    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", default="outputs/events")
    parser.add_argument("--pattern", default="reconstructed_2024-01-*.jsonl")
    parser.add_argument("--speedup-report", default="outputs/reports/step3_speedup.json")
    parser.add_argument("--report-output", default="outputs/reports/tier2_ippo_report.json")
    parser.add_argument("--checkpoint-output", default="outputs/checkpoints/tier2_ippo_params.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = LOCKED_IPPO_CONFIG
    if args.batch_size != 64:
        raise SystemExit("Tier 2 batch size must be N=64")
    if args.agents != config.agents_per_environment:
        raise SystemExit("Tier 2 requires exactly 2 agents")

    event_files = sorted(_project_path(args.events_dir).glob(args.pattern))
    if not event_files:
        raise SystemExit("no reconstructed event files found for Tier 2")
    events = []
    for path in event_files:
        events.extend(read_events_jsonl(path))
    windows = slice_non_overlapping_windows(events, episode_length_ms=EPISODE_LENGTH_MS)
    if not windows:
        raise SystemExit("no usable episode windows found")

    flow_signals = compute_informed_flow_signals(events)
    proxy_fire_count = sum(1 for signal in flow_signals.values() if signal.informed_heavy)
    flow_series = flow_value_series(flow_signals, len(events) + 1)
    speedup = _load_json_if_exists(_project_path(args.speedup_report))
    _assert_batched_jax_path_compiles()

    params = init_two_agent_params(args.seed)
    adam_states = tuple(init_adam(agent_params) for agent_params in params)
    rng = jax.random.PRNGKey(args.seed)
    initial_state = make_batched_state(windows, args.batch_size, args.agents)
    initial_obs = np.asarray(batched_observations(initial_state, informed_flow_value=current_batched_flow_values(initial_state, flow_series)))
    initial_summaries = [action_summary(agent_params, initial_obs[:, agent_idx, :]) for agent_idx, agent_params in enumerate(params)]

    agent_losses: list[list[float]] = [[] for _ in range(args.agents)]
    action_drift: list[dict[str, float]] = []
    updates = max(1, args.training_steps // config.rollout_length)
    total_bid_overflows = 0
    total_ask_overflows = 0
    for _ in range(updates):
        rng, rollout_key = jax.random.split(rng)
        rollout, rollout_bid_overflows, rollout_ask_overflows = _collect_rollout(
            windows, args.batch_size, args.agents, config.rollout_length, params, rollout_key, flow_series
        )
        total_bid_overflows += rollout_bid_overflows
        total_ask_overflows += rollout_ask_overflows
        for agent_idx in range(args.agents):
            obs = jnp.asarray(rollout["obs"][:, agent_idx, :], dtype=jnp.float32)
            actions = jnp.asarray(rollout["actions"][:, agent_idx, :], dtype=jnp.float32)
            old_log_probs = jnp.asarray(rollout["old_log_probs"][:, agent_idx], dtype=jnp.float32)
            values = jnp.asarray(rollout["values"][:, agent_idx], dtype=jnp.float32)
            rewards = jnp.asarray(rollout["rewards"][:, agent_idx], dtype=jnp.float32)
            returns = rewards
            advantages = returns - values
            batch = {
                "obs": obs,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "returns": returns,
                "advantages": (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8),
            }
            for _ in range(config.update_epochs):
                params_list = list(params)
                adam_list = list(adam_states)
                updated_params, updated_adam, loss = ppo_update(params[agent_idx], adam_states[agent_idx], batch, config)
                params_list[agent_idx] = updated_params
                adam_list[agent_idx] = updated_adam
                params = tuple(params_list)
                adam_states = tuple(adam_list)
            agent_losses[agent_idx].append(float(loss))

    final_summaries = [action_summary(agent_params, initial_obs[:, agent_idx, :]) for agent_idx, agent_params in enumerate(params)]
    for before, after in zip(initial_summaries, final_summaries):
        action_drift.append(
            {
                "offset_mean_abs_change": abs(after["offset_mean"] - before["offset_mean"]),
                "quantity_mean_abs_change": abs(after["quantity_mean"] - before["quantity_mean"]),
            }
        )

    _write_checkpoint(_project_path(args.checkpoint_output), params)
    report = {
        "event_files": [str(path.relative_to(ROOT)) for path in event_files],
        "batch_size": args.batch_size,
        "agents": args.agents,
        "training_steps_requested": args.training_steps,
        "rollout_execution_path": "env.jax_batched_env",
        "rollout_length": config.rollout_length,
        "updates": updates,
        "ippo_config": config.__dict__,
        "speedup_report": speedup,
        "speedup_ratio_available": _speedup_ratio_available(speedup),
        "informed_flow_proxy_fire_count": proxy_fire_count,
        "capacity_overflows": {
            "bid": total_bid_overflows,
            "ask": total_ask_overflows,
            "total": total_bid_overflows + total_ask_overflows,
        },
        "capacity_overflow_occurred": (total_bid_overflows + total_ask_overflows) > 0,
        "initial_action_summaries": initial_summaries,
        "final_action_summaries": final_summaries,
        "action_drift": action_drift,
        "agent_losses": agent_losses,
        "checkpoint_output": str(_project_path(args.checkpoint_output).relative_to(ROOT)),
        "acceptance": {
            "speedup_ratio_real_and_recorded": _speedup_ratio_available(speedup),
            "step7_proxy_fired_at_least_once": proxy_fire_count > 0,
            "policies_diverged_from_initialization": all(
                item["offset_mean_abs_change"] > 0.0 or item["quantity_mean_abs_change"] > 0.0 for item in action_drift
            ),
        },
    }
    output = _project_path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["capacity_overflow_occurred"]:
        overflows = report["capacity_overflows"]
        print(
            "WARNING: capacity overflow occurred during training "
            f"({overflows['total']} dropped orders, {overflows['bid']} bid / {overflows['ask']} ask). "
            "This violates the validated-data assumption and should never happen for data that passed validation mode.",
            file=sys.stderr,
        )


def _collect_rollout(windows, batch_size: int, agents: int, rollout_length: int, params, rng, flow_series) -> tuple[dict[str, np.ndarray], int, int]:
    import jax
    import jax.numpy as jnp

    state = make_batched_state(windows, batch_size, agents)
    obs_rows = []
    action_rows = []
    log_prob_rows = []
    value_rows = []
    reward_rows = []
    for _ in range(rollout_length):
        obs = batched_observations(state, informed_flow_value=current_batched_flow_values(state, flow_series))
        action_cols = []
        log_prob_cols = []
        value_cols = []
        for agent_idx in range(agents):
            rng, action_key = jax.random.split(rng)
            action, log_prob, value = sample_action(params[agent_idx], obs[:, agent_idx, :], action_key)
            action_cols.append(action)
            log_prob_cols.append(log_prob)
            value_cols.append(value)
        actions = jnp.stack(action_cols, axis=1)
        state, rewards = batched_step(state, actions)
        obs_rows.append(np.asarray(obs))
        action_rows.append(np.asarray(actions))
        log_prob_rows.append(np.asarray(jnp.stack(log_prob_cols, axis=1)))
        value_rows.append(np.asarray(jnp.stack(value_cols, axis=1)))
        reward_rows.append(np.asarray(rewards))
    return (
        {
            "obs": np.asarray(obs_rows, dtype=np.float32).reshape((-1, agents, 90)),
            "actions": np.asarray(action_rows, dtype=np.float32).reshape((-1, agents, 2)),
            "old_log_probs": np.asarray(log_prob_rows, dtype=np.float32).reshape((-1, agents)),
            "values": np.asarray(value_rows, dtype=np.float32).reshape((-1, agents)),
            "rewards": np.asarray(reward_rows, dtype=np.float32).reshape((-1, agents)),
        },
        int(np.asarray(state.bid_overflows, dtype=np.int64).sum()),
        int(np.asarray(state.ask_overflows, dtype=np.int64).sum()),
    )


def _assert_batched_jax_path_compiles() -> None:
    import jax.numpy as jnp

    bids = jnp.zeros((2, 4, 5), dtype=jnp.float32).at[:, :, 0].set(-1.0)
    asks = jnp.zeros((2, 4, 5), dtype=jnp.float32).at[:, :, 0].set(-1.0)
    events = jnp.asarray([[0, 0, 100.0, 1.0, 1.0, 0.0, 0.0], [0, 1, 101.0, 1.0, 2.0, 0.0, 0.0]], dtype=jnp.float32)
    out_bids, _, _, _, _ = batched_apply_event_arrays(bids, asks, events)
    out_bids.block_until_ready()


def _write_checkpoint(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for agent_idx, agent_params in enumerate(params):
        leaves, _ = __import__("jax").tree_util.tree_flatten(agent_params)
        for leaf_idx, leaf in enumerate(leaves):
            flat[f"agent{agent_idx}_leaf{leaf_idx}"] = np.asarray(leaf)
    np.savez(path, **flat)


def _speedup_ratio_available(speedup: dict | None) -> bool:
    if not speedup:
        return False
    measurements = speedup.get("measurements", [])
    required = {16, 64, 256}
    seen = {int(row.get("batch_size")) for row in measurements if row.get("speedup_ratio") is not None}
    return required.issubset(seen)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
