from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on machines without JAX.
    jax = None
    jnp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from lob.constants import EMPTY_PRICE, EVENT_ADD, EVENT_CANCEL, EVENT_EXECUTE, SIDE_ASK, SIDE_BID


def require_jax() -> None:
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("JAX is required for lob.jax_matching") from _IMPORT_ERROR


if jnp is not None:

    def empty_books(batch_size: int, capacity: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        bids = jnp.zeros((batch_size, capacity, 5), dtype=jnp.float32).at[:, :, 0].set(EMPTY_PRICE)
        asks = jnp.zeros((batch_size, capacity, 5), dtype=jnp.float32).at[:, :, 0].set(EMPTY_PRICE)
        return bids, asks


    @jax.jit
    def batched_apply_event_arrays(
        bids: jnp.ndarray,
        asks: jnp.ndarray,
        events: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return jax.vmap(_apply_one_event)(bids, asks, events)


    def batched_top_levels(bids: jnp.ndarray, asks: jnp.ndarray, levels: int = 20) -> tuple[jnp.ndarray, jnp.ndarray]:
        bid_levels = jax.vmap(lambda side: _top_levels_one(side, SIDE_BID, levels))(bids)
        ask_levels = jax.vmap(lambda side: _top_levels_one(side, SIDE_ASK, levels))(asks)
        return bid_levels, ask_levels


    def _apply_one_event(
        bids: jnp.ndarray,
        asks: jnp.ndarray,
        event: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        event_type = event[0].astype(jnp.int32)
        side_code = event[1].astype(jnp.int32)
        price = event[2]
        quantity = event[3]
        order_id = event[4]
        trader_id = event[5]
        timestamp = event[6]

        def add_case(state):
            bid_state, ask_state = state
            row = jnp.array([price, quantity, order_id, trader_id, timestamp], dtype=bid_state.dtype)
            bid_out, bid_drop = jax.lax.cond(
                side_code == SIDE_BID,
                lambda arr: _add_one(arr, row),
                lambda arr: (arr, jnp.asarray(0.0, dtype=bid_state.dtype)),
                bid_state,
            )
            ask_out, ask_drop = jax.lax.cond(
                side_code == SIDE_ASK,
                lambda arr: _add_one(arr, row),
                lambda arr: (arr, jnp.asarray(0.0, dtype=ask_state.dtype)),
                ask_state,
            )
            return bid_out, ask_out, jnp.array(0.0, dtype=bid_state.dtype), bid_drop, ask_drop

        def cancel_case(state):
            bid_state, ask_state = state
            bid_out = jax.lax.cond(
                side_code == SIDE_BID,
                lambda arr: _cancel_one(arr, price, quantity, order_id),
                lambda arr: arr,
                bid_state,
            )
            ask_out = jax.lax.cond(
                side_code == SIDE_ASK,
                lambda arr: _cancel_one(arr, price, quantity, order_id),
                lambda arr: arr,
                ask_state,
            )
            return (
                bid_out,
                ask_out,
                jnp.array(0.0, dtype=bid_state.dtype),
                jnp.array(0.0, dtype=bid_state.dtype),
                jnp.array(0.0, dtype=ask_state.dtype),
            )

        def execute_case(state):
            bid_state, ask_state = state
            bid_out, bid_fill = jax.lax.cond(
                side_code == SIDE_BID,
                lambda arr: _execute_exact_one(arr, price, quantity),
                lambda arr: (arr, jnp.array(0.0, dtype=arr.dtype)),
                bid_state,
            )
            ask_out, ask_fill = jax.lax.cond(
                side_code == SIDE_ASK,
                lambda arr: _execute_exact_one(arr, price, quantity),
                lambda arr: (arr, jnp.array(0.0, dtype=arr.dtype)),
                ask_state,
            )
            return (
                bid_out,
                ask_out,
                bid_fill + ask_fill,
                jnp.array(0.0, dtype=bid_state.dtype),
                jnp.array(0.0, dtype=ask_state.dtype),
            )

        return jax.lax.switch(event_type, (add_case, cancel_case, execute_case), (bids, asks))


    def _add_one(arr: jnp.ndarray, row: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        empty = arr[:, 0] == EMPTY_PRICE
        insert_idx = jnp.argmax(empty)
        has_empty = jnp.any(empty)
        return jax.lax.cond(
            has_empty,
            lambda side: (side.at[insert_idx].set(row), jnp.asarray(0.0, dtype=arr.dtype)),
            lambda side: (side, jnp.asarray(1.0, dtype=arr.dtype)),
            arr,
        )


    def _cancel_fifo_one(arr: jnp.ndarray, price: jnp.ndarray, quantity: jnp.ndarray) -> jnp.ndarray:
        def body(carry, _):
            side, remaining = carry
            rows = (side[:, 0] == price) & (side[:, 1] > 0)
            sort_key = jnp.where(rows, side[:, 4], jnp.inf)
            row = jnp.argmin(sort_key)
            can_take = rows[row] & (remaining > 1e-12)
            take = jnp.where(can_take, jnp.minimum(remaining, side[row, 1]), 0.0)
            updated_qty = side[row, 1] - take
            updated = side.at[row, 1].set(updated_qty)
            cleared = updated.at[row].set(jnp.array([EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0], dtype=side.dtype))
            side = jax.lax.cond(can_take & (updated_qty <= 1e-12), lambda _: cleared, lambda _: updated, operand=None)
            return (side, remaining - take), None

        (out, _), _ = jax.lax.scan(body, (arr, quantity), xs=None, length=arr.shape[0])
        return out


    def _cancel_one(arr: jnp.ndarray, price: jnp.ndarray, quantity: jnp.ndarray, order_id: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.cond(
            order_id != 0,
            lambda side: _cancel_by_order_id_one(side, order_id, quantity),
            lambda side: _cancel_fifo_one(side, price, quantity),
            arr,
        )


    def _cancel_by_order_id_one(arr: jnp.ndarray, order_id: jnp.ndarray, quantity: jnp.ndarray) -> jnp.ndarray:
        rows = (arr[:, 0] != EMPTY_PRICE) & (arr[:, 2] == order_id)
        row = jnp.argmax(rows)
        can_take = jnp.any(rows)
        take = jnp.where(can_take, jnp.minimum(quantity, arr[row, 1]), 0.0)
        updated_qty = arr[row, 1] - take
        updated = arr.at[row, 1].set(updated_qty)
        cleared = updated.at[row].set(jnp.array([EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0], dtype=arr.dtype))
        return jax.lax.cond(can_take & (updated_qty <= 1e-12), lambda _: cleared, lambda _: updated, operand=None)


    def _execute_exact_one(arr: jnp.ndarray, price: jnp.ndarray, quantity: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        before = jnp.sum(arr[:, 1])
        out = _cancel_fifo_one(arr, price, quantity)
        after = jnp.sum(out[:, 1])
        return out, before - after


    def _top_levels_one(arr: jnp.ndarray, side_code: int, levels: int) -> jnp.ndarray:
        price = arr[:, 0]
        quantity = arr[:, 1]
        active = price != EMPTY_PRICE
        price_key = jnp.where(side_code == SIDE_BID, -price, price)
        price_key = jnp.where(active, price_key, jnp.inf)
        order = jnp.argsort(price_key, stable=True)
        selected = order[:levels]
        out = jnp.stack([price[selected], quantity[selected]], axis=1)
        return jnp.where(out[:, :1] == EMPTY_PRICE, jnp.nan, out)
else:

    def empty_books(batch_size: int, capacity: int):
        require_jax()


    def batched_apply_event_arrays(bids, asks, events):
        require_jax()


    def batched_top_levels(bids, asks, levels: int = 20):
        require_jax()
