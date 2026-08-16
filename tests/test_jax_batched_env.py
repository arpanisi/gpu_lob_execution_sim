from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from env.jax_batched_env import require_jax_batched_env


def test_jax_batched_env_guard_is_explicit_when_jax_is_missing() -> None:
    if importlib.util.find_spec("jax") is not None:
        require_jax_batched_env()
    else:
        with pytest.raises(ModuleNotFoundError):
            require_jax_batched_env()


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_batched_env_produces_batched_observations_and_rewards() -> None:
    import jax.numpy as jnp

    from env.jax_batched_env import batched_observations, batched_step, make_batched_state
    from env.windows import slice_non_overlapping_windows
    from lob.events import BookEvent

    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
        BookEvent("CANCEL", "ask", 101.0, 0.1, 0, 0, 1001),
    ]
    windows = slice_non_overlapping_windows(events, episode_length_ms=600_000)
    state = make_batched_state(windows, batch_size=4, agents=2)
    obs = batched_observations(state)
    next_state, rewards = batched_step(state, jnp.zeros((4, 2, 2), dtype=jnp.float32))

    assert obs.shape == (4, 2, 89)
    assert rewards.shape == (4, 2)
    assert next_state.bids.shape == (4, 1000, 5)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_batched_state_tracks_per_environment_overflow_and_flow_offsets() -> None:
    from env.jax_batched_env import make_batched_state
    from env.windows import slice_non_overlapping_windows
    from lob.events import BookEvent

    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
        BookEvent("CANCEL", "ask", 101.0, 0.1, 0, 0, 1001),
    ]
    windows = slice_non_overlapping_windows(events, episode_length_ms=600_000)
    state = make_batched_state(windows, batch_size=4, agents=2)

    assert state.bid_overflows.shape == (4,)
    assert state.ask_overflows.shape == (4,)
    assert state.flow_offsets.shape == (4,)
    assert int(np.asarray(state.bid_overflows).sum()) == 0
    assert int(np.asarray(state.ask_overflows).sum()) == 0


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_batched_historical_add_overflow_increments_per_environment_counter() -> None:
    import jax.numpy as jnp

    from env.episode import make_window_from_events
    from env.jax_batched_env import batched_step, make_batched_state
    from lob.constants import CAPACITY
    from lob.events import BookEvent

    events = [BookEvent("ADD", "bid", 100.0, 1.0, order_id + 1, 0, 0) for order_id in range(CAPACITY)]
    events.append(BookEvent("ADD", "ask", 101.0, 1.0, 2001, 0, 0))
    events.append(BookEvent("ADD", "bid", 98.0, 1.0, 3001, 0, 1))

    window = make_window_from_events(events)
    state = make_batched_state([window], batch_size=1, agents=2)
    next_state, _ = batched_step(state, jnp.zeros((1, 2, 2), dtype=jnp.float32))

    assert int(np.asarray(next_state.bid_overflows)[0]) == 1
    assert int(np.asarray(next_state.ask_overflows)[0]) == 0


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_batched_agent_remainder_overflow_increments_counter() -> None:
    import jax.numpy as jnp

    from env.jax_batched_env import _add_agent_remainder_batch
    from lob.constants import AGENT_ID_START, CAPACITY, EMPTY_PRICE

    arr = jnp.zeros((2, CAPACITY, 5), dtype=jnp.float32).at[:, :, 0].set(EMPTY_PRICE)
    arr = arr.at[:, :, 0].set(100.0)
    should_add = jnp.asarray([True, True])
    quantity = jnp.asarray([0.5, 0.5])
    price = jnp.asarray([99.0, 99.0])
    timestamp = jnp.asarray([10.0, 10.0])

    out, dropped = _add_agent_remainder_batch(arr, should_add, price, quantity, AGENT_ID_START, timestamp)
    assert out.shape == arr.shape
    assert [float(item) for item in dropped] == [1.0, 1.0]


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_batched_observations_use_per_environment_informed_flow_value() -> None:
    import jax.numpy as jnp

    from env.jax_batched_env import batched_observations, make_batched_state
    from env.windows import slice_non_overlapping_windows
    from lob.events import BookEvent

    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
        BookEvent("CANCEL", "ask", 101.0, 0.1, 0, 0, 1001),
    ]
    windows = slice_non_overlapping_windows(events, episode_length_ms=600_000)
    state = make_batched_state(windows, batch_size=4, agents=2)
    flow_values = jnp.asarray([1.5, -2.5, 3.25, 0.0], dtype=jnp.float32)

    obs = batched_observations(state, informed_flow_value=flow_values)

    assert obs.shape == (4, 2, 89)
    for env in range(4):
        for agent in range(2):
            assert float(obs[env, agent, 86]) == pytest.approx(float(flow_values[env]))


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_current_batched_flow_values_maps_cursor_to_flat_replay_position() -> None:
    import jax.numpy as jnp

    from data.informed_flow import compute_informed_flow_signals, current_flow_value, flow_value_series
    from env.jax_batched_env import batched_step, current_batched_flow_values, make_batched_state
    from env.windows import slice_non_overlapping_windows
    from lob.events import BookEvent

    events = [
        BookEvent("ADD", "bid", 100.0, 5.0, 1, 0, 0),
        BookEvent("ADD", "ask", 101.0, 5.0, 2, 0, 0),
    ]
    order_id = 3
    for idx in range(40):
        timestamp = idx + 1
        events.append(BookEvent("EXECUTE", "ask", 101.0, 0.1, 0, 90, timestamp))
        events.append(BookEvent("ADD", "ask", 101.0, 0.1, order_id, 0, timestamp))
        order_id += 1
        events.append(BookEvent("ADD", "bid", 100.0 + idx * 0.02, 0.5, order_id, 0, timestamp))
        order_id += 1

    signals = compute_informed_flow_signals(events, horizon_events=10)
    series = flow_value_series(signals, len(events) + 1)
    windows = slice_non_overlapping_windows(events, episode_length_ms=600_000)
    state = make_batched_state(windows, batch_size=4, agents=2)
    offsets = np.asarray(state.flow_offsets)

    values = current_batched_flow_values(state, series)
    for env in range(4):
        expected = current_flow_value(signals, int(offsets[env]))
        assert float(values[env]) == pytest.approx(expected)

    next_state, _ = batched_step(state, jnp.zeros((4, 2, 2), dtype=jnp.float32))
    cursors = np.asarray(next_state.cursors)
    next_values = current_batched_flow_values(next_state, series)
    for env in range(4):
        expected = current_flow_value(signals, int(offsets[env]) + int(cursors[env]))
        assert float(next_values[env]) == pytest.approx(expected)
