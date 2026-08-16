from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.observation import OBSERVATION_LENGTH
from lob.constants import AGENT_ID_START, CAPACITY, EMPTY_PRICE, EVENT_CANCEL, SIDE_BID, STEP_EVENT_COUNT, TICK_SIZE
from lob.matching import event_to_array
from lob.jax_matching import batched_apply_event_arrays, batched_top_levels, require_jax

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    jax = None
    jnp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class JaxBatchedState:
    bids: object
    asks: object
    events: object
    cursors: object
    start_ts: object
    end_ts: object
    current_ts: object
    arrival_mid: object
    remaining: object
    executed: object
    task_sides: object
    done: object
    bid_overflows: object
    ask_overflows: object
    flow_offsets: object


def require_jax_batched_env() -> None:
    require_jax()
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("JAX is required for env.jax_batched_env") from _IMPORT_ERROR


if jnp is not None:

    def make_batched_state(windows, batch_size: int, agents: int = 2, target_quantity: float = 0.05) -> JaxBatchedState:
        if agents != 2:
            raise ValueError("batched JAX training environment is locked to exactly 2 agents")
        selected = [windows[idx % len(windows)] for idx in range(batch_size)]
        max_events = max(len(window.events) for window in selected)
        bids = np.stack([window.initial_book.bids for window in selected]).astype(np.float32)
        asks = np.stack([window.initial_book.asks for window in selected]).astype(np.float32)
        events = np.zeros((batch_size, max_events, 7), dtype=np.float32)
        events[:, :, 0] = EVENT_CANCEL
        events[:, :, 1] = SIDE_BID
        for env_idx, window in enumerate(selected):
            if window.events:
                events[env_idx, : len(window.events)] = np.stack([event_to_array(event) for event in window.events]).astype(np.float32)
        bid0, ask0 = _best_bid_ask(jnp.asarray(bids), jnp.asarray(asks))
        arrival_mid = (bid0 + ask0) / 2.0
        remaining = jnp.full((batch_size, agents), float(target_quantity), dtype=jnp.float32)
        executed = jnp.zeros((batch_size, agents), dtype=jnp.float32)
        task_sides = jnp.tile(jnp.asarray([1.0, -1.0], dtype=jnp.float32), (batch_size, 1))
        return JaxBatchedState(
            bids=jnp.asarray(bids),
            asks=jnp.asarray(asks),
            events=jnp.asarray(events),
            cursors=jnp.zeros((batch_size,), dtype=jnp.int32),
            start_ts=jnp.asarray([window.start_ts for window in selected], dtype=jnp.float32),
            end_ts=jnp.asarray([window.end_ts for window in selected], dtype=jnp.float32),
            current_ts=jnp.asarray([window.start_ts for window in selected], dtype=jnp.float32),
            arrival_mid=arrival_mid,
            remaining=remaining,
            executed=executed,
            task_sides=task_sides,
            done=jnp.zeros((batch_size,), dtype=bool),
            bid_overflows=jnp.zeros((batch_size,), dtype=jnp.int32),
            ask_overflows=jnp.zeros((batch_size,), dtype=jnp.int32),
            flow_offsets=jnp.asarray([window.flat_event_offset for window in selected], dtype=jnp.int32),
        )


    def batched_step(state: JaxBatchedState, actions: jnp.ndarray, k_events: int = STEP_EVENT_COUNT) -> tuple[JaxBatchedState, jnp.ndarray]:
        bids = state.bids
        asks = state.asks
        remaining = state.remaining
        executed = state.executed
        rewards = jnp.zeros_like(remaining)
        bid_overflows = state.bid_overflows
        ask_overflows = state.ask_overflows
        bids, asks = _cancel_all_agent_orders(bids, asks)

        for agent_idx in range(2):
            bids, asks, remaining, executed, agent_reward, agent_bid_ovf, agent_ask_ovf = _apply_agent_actions(
                bids,
                asks,
                actions[:, agent_idx, :],
                remaining,
                executed,
                state.task_sides[:, agent_idx],
                state.arrival_mid,
                agent_idx,
                state.current_ts + 1.0,
            )
            bid_overflows = bid_overflows + agent_bid_ovf.astype(jnp.int32)
            ask_overflows = ask_overflows + agent_ask_ovf.astype(jnp.int32)
            rewards = rewards.at[:, agent_idx].set(agent_reward)

        def replay_one(carry, offset):
            replay_bids, replay_asks, acc_bid_ovf, acc_ask_ovf = carry
            indices = jnp.minimum(state.cursors + offset, state.events.shape[1] - 1)
            event_batch = state.events[jnp.arange(state.events.shape[0]), indices]
            valid = (state.cursors + offset) < state.events.shape[1]
            noop = jnp.asarray([EVENT_CANCEL, SIDE_BID, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
            event_batch = jnp.where(valid[:, None], event_batch, noop[None, :])
            replay_bids, replay_asks, _, bid_drop, ask_drop = batched_apply_event_arrays(replay_bids, replay_asks, event_batch)
            return (replay_bids, replay_asks, acc_bid_ovf + bid_drop.astype(jnp.int32), acc_ask_ovf + ask_drop.astype(jnp.int32)), None

        (bids, asks, bid_overflows, ask_overflows), _ = jax.lax.scan(
            replay_one, (bids, asks, bid_overflows, ask_overflows), jnp.arange(k_events)
        )
        new_cursors = jnp.minimum(state.cursors + k_events, state.events.shape[1])
        next_events = state.events[jnp.arange(state.events.shape[0]), jnp.maximum(new_cursors - 1, 0)]
        current_ts = jnp.maximum(state.current_ts, next_events[:, 6])
        done = state.done | (new_cursors >= state.events.shape[1]) | (current_ts >= state.end_ts)
        next_state = JaxBatchedState(
            bids=bids,
            asks=asks,
            events=state.events,
            cursors=new_cursors,
            start_ts=state.start_ts,
            end_ts=state.end_ts,
            current_ts=current_ts,
            arrival_mid=state.arrival_mid,
            remaining=remaining,
            executed=executed,
            task_sides=state.task_sides,
            done=done,
            bid_overflows=bid_overflows,
            ask_overflows=ask_overflows,
            flow_offsets=state.flow_offsets,
        )
        return next_state, rewards


    def batched_observations(state: JaxBatchedState, informed_flow_value: object = None):
        if informed_flow_value is None:
            informed_flow_value = jnp.zeros((state.bids.shape[0],), dtype=jnp.float32)
        informed_flow_value = jnp.asarray(informed_flow_value, dtype=jnp.float32)
        if informed_flow_value.shape[0] != state.bids.shape[0]:
            raise ValueError("informed_flow_value must be a per-environment array of shape (batch_size,)")
        bid_levels, ask_levels = batched_top_levels(state.bids, state.asks, levels=20)
        bid_best, ask_best = _best_bid_ask(state.bids, state.asks)
        mid = (bid_best + ask_best) / 2.0
        elapsed = jnp.clip((state.current_ts - state.start_ts) / jnp.maximum(1.0, state.end_ts - state.start_ts), 0.0, 1.0)
        obs = []
        for agent_idx in range(2):
            own_price, own_qty = _agent_resting_order_batch(state.bids, state.asks, AGENT_ID_START + agent_idx, state.arrival_mid)
            extras = jnp.stack(
                [
                    mid,
                    elapsed,
                    state.remaining[:, agent_idx] / jnp.maximum(1e-12, state.remaining[:, agent_idx] + state.executed[:, agent_idx]),
                    state.executed[:, agent_idx] / jnp.maximum(1e-12, state.remaining[:, agent_idx] + state.executed[:, agent_idx]),
                    state.arrival_mid,
                    mid - state.arrival_mid,
                    informed_flow_value,
                    own_price,
                    own_qty,
                ],
                axis=1,
            )
            flat = jnp.concatenate([bid_levels.reshape((bid_levels.shape[0], 40)), ask_levels.reshape((ask_levels.shape[0], 40)), extras], axis=1)
            obs.append(jnp.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0))
        stacked = jnp.stack(obs, axis=1)
        if stacked.shape[-1] != OBSERVATION_LENGTH:
            raise RuntimeError(f"observation length {stacked.shape[-1]} != {OBSERVATION_LENGTH}")
        return stacked


    def current_batched_flow_values(state: JaxBatchedState, flow_series: list[float]) -> jnp.ndarray:
        """Per-environment current informed-flow values for each env's own replay position.

        Each environment's cursor is a position into its own sliced event array, but
        ``compute_informed_flow_signals`` was computed over the flat event list; ``flow_offsets``
        maps each window's replayable slice back into that flat index space so ``flow_series`` can
        be indexed directly (positions never exceed the series length).
        """
        positions = np.asarray(state.flow_offsets, dtype=np.int64) + np.asarray(state.cursors, dtype=np.int64)
        positions = np.minimum(positions, len(flow_series) - 1)
        return jnp.asarray(np.take(flow_series, positions), dtype=jnp.float32)


    def _apply_agent_actions(bids, asks, actions, remaining, executed, task_side, arrival_mid, agent_idx: int, timestamp):
        quantity = jnp.clip(actions[:, 1], 0.0, remaining[:, agent_idx])
        bid_best, ask_best = _best_bid_ask(bids, asks)
        limit_price = jnp.where(task_side > 0, ask_best + actions[:, 0] * TICK_SIZE, bid_best - actions[:, 0] * TICK_SIZE)
        trader_id = AGENT_ID_START + agent_idx

        asks, buy_filled_qty, buy_cost = _match_batch(asks, "ask", limit_price, quantity, trader_id, timestamp)
        bids, sell_filled_qty, sell_cost = _match_batch(bids, "bid", limit_price, quantity, trader_id, timestamp)
        filled_qty = jnp.where(task_side > 0, buy_filled_qty, sell_filled_qty)
        gross = jnp.where(task_side > 0, buy_cost, sell_cost)
        unfilled = jnp.maximum(0.0, quantity - filled_qty)
        bids, bid_overflow = _add_agent_remainder_batch(bids, task_side > 0, limit_price, unfilled, trader_id, timestamp)
        asks, ask_overflow = _add_agent_remainder_batch(asks, task_side < 0, limit_price, unfilled, trader_id, timestamp)
        cost = jnp.where(task_side > 0, gross - arrival_mid * filled_qty, arrival_mid * filled_qty - gross)
        initial_target = jnp.maximum(1e-12, remaining[:, agent_idx] + executed[:, agent_idx])
        reward = -cost / (initial_target * arrival_mid)
        remaining = remaining.at[:, agent_idx].set(jnp.maximum(0.0, remaining[:, agent_idx] - filled_qty))
        executed = executed.at[:, agent_idx].set(executed[:, agent_idx] + filled_qty)
        return bids, asks, remaining, executed, reward, bid_overflow, ask_overflow


    def _match_batch(side_arr, side_name: str, limit_price, quantity, trader_id: int, timestamp):
        return jax.vmap(lambda arr, price, qty, ts: _match_one(arr, side_name, price, qty, trader_id, ts))(side_arr, limit_price, quantity, timestamp)


    def _match_one(arr, side_name: str, limit_price, quantity, trader_id: int, timestamp):
        def body(carry, _):
            side, remaining, filled, gross = carry
            active = (side[:, 0] != EMPTY_PRICE) & (side[:, 1] > 0)
            if side_name == "ask":
                marketable = active & (side[:, 0] <= limit_price)
                price_key = side[:, 0]
            else:
                marketable = active & (side[:, 0] >= limit_price)
                price_key = -side[:, 0]
            sort_key = jnp.where(marketable, price_key * 1_000_000_000.0 + side[:, 4], jnp.inf)
            row = jnp.argmin(sort_key)
            can_take = marketable[row] & (remaining > 1e-12)
            take = jnp.where(can_take, jnp.minimum(remaining, side[row, 1]), 0.0)
            updated_qty = side[row, 1] - take
            updated = side.at[row, 1].set(updated_qty)
            cleared = updated.at[row].set(jnp.asarray([EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0], dtype=side.dtype))
            side = jax.lax.cond(can_take & (updated_qty <= 1e-12), lambda _: cleared, lambda _: updated, operand=None)
            return (side, remaining - take, filled + take, gross + take * side[row, 0]), None

        (out, _, filled, gross), _ = jax.lax.scan(body, (arr, quantity, jnp.array(0.0, arr.dtype), jnp.array(0.0, arr.dtype)), xs=None, length=CAPACITY)
        return out, filled, gross


    def _add_agent_remainder_batch(arr, should_add, price, quantity, trader_id: int, timestamp):
        return jax.vmap(lambda side, flag, px, qty, ts: _add_agent_remainder_one(side, flag, px, qty, trader_id, ts))(arr, should_add, price, quantity, timestamp)


    def _add_agent_remainder_one(arr, should_add, price, quantity, trader_id: int, timestamp):
        empty = arr[:, 0] == EMPTY_PRICE
        idx = jnp.argmax(empty)
        can_add = should_add & (quantity > 1e-12) & jnp.any(empty)
        dropped = should_add & (quantity > 1e-12) & (~jnp.any(empty))
        order_id = float(trader_id) * 1_000_000_000_000.0 + timestamp
        row = jnp.asarray([price, quantity, order_id, float(trader_id), timestamp], dtype=arr.dtype)
        return jax.lax.cond(
            can_add,
            lambda side: (side.at[idx].set(row), jnp.asarray(0.0, dtype=arr.dtype)),
            lambda side: (side, jnp.asarray(dropped, dtype=arr.dtype)),
            arr,
        )


    def _cancel_all_agent_orders(bids, asks):
        for agent_idx in range(2):
            trader_id = float(AGENT_ID_START + agent_idx)
            bids = jnp.where((bids[:, :, 3:4] == trader_id), jnp.asarray([EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0], dtype=bids.dtype), bids)
            asks = jnp.where((asks[:, :, 3:4] == trader_id), jnp.asarray([EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0], dtype=asks.dtype), asks)
        return bids, asks


    def _best_bid_ask(bids, asks):
        bid_price = jnp.where(bids[:, :, 0] != EMPTY_PRICE, bids[:, :, 0], -jnp.inf)
        ask_price = jnp.where(asks[:, :, 0] != EMPTY_PRICE, asks[:, :, 0], jnp.inf)
        return jnp.max(bid_price, axis=1), jnp.min(ask_price, axis=1)


    def _agent_resting_order_batch(bids, asks, trader_id: int, arrival_mid):
        bid_match = (bids[:, :, 0] != EMPTY_PRICE) & (bids[:, :, 3] == float(trader_id))
        ask_match = (asks[:, :, 0] != EMPTY_PRICE) & (asks[:, :, 3] == float(trader_id))
        bid_idx = jnp.argmax(bid_match, axis=1)
        ask_idx = jnp.argmax(ask_match, axis=1)
        has_bid = jnp.any(bid_match, axis=1)
        has_ask = jnp.any(ask_match, axis=1)
        bid_price = bids[jnp.arange(bids.shape[0]), bid_idx, 0]
        ask_price = asks[jnp.arange(asks.shape[0]), ask_idx, 0]
        bid_qty = bids[jnp.arange(bids.shape[0]), bid_idx, 1]
        ask_qty = asks[jnp.arange(asks.shape[0]), ask_idx, 1]
        price = jnp.where(has_bid, bid_price - arrival_mid, jnp.where(has_ask, ask_price - arrival_mid, -1.0))
        qty = jnp.where(has_bid, bid_qty, jnp.where(has_ask, ask_qty, 0.0))
        return price, qty
else:

    def make_batched_state(windows, batch_size: int, agents: int = 2, target_quantity: float = 0.05):
        require_jax_batched_env()


    def batched_step(state, actions, k_events: int = STEP_EVENT_COUNT):
        require_jax_batched_env()


    def batched_observations(state, informed_flow_value=None):
        require_jax_batched_env()
