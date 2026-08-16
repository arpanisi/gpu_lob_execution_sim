from __future__ import annotations

from dataclasses import dataclass, field

from lob.constants import MARKET_ID
from lob.events import BookEvent, DepthRecord, TradePrint, expected_aggressor_for_decrease, market_add


@dataclass
class ReconstructionResult:
    events: list[BookEvent]
    trade_consumed: dict[str, float]


@dataclass
class _TradeState:
    trade: TradePrint
    remaining: float


def reconstruct_events(depth_records: list[DepthRecord], trades: list[TradePrint], *, tolerance_ms: int = 1_000) -> ReconstructionResult:
    if not depth_records:
        return ReconstructionResult([], {})
    ordered_depth = sorted(depth_records, key=lambda record: record.timestamp)
    trade_states = [_TradeState(trade, float(trade.quantity)) for trade in sorted(trades, key=lambda item: item.timestamp)]
    reference: dict[tuple[str, float], float] = {}
    events: list[BookEvent] = []
    next_order_id = 1

    first = ordered_depth[0]
    for side, levels in (("bid", first.bids), ("ask", first.asks)):
        for price, quantity in levels:
            if quantity <= 0:
                continue
            events.append(market_add(side, price, quantity, next_order_id, first.timestamp))
            reference[(side, float(price))] = float(quantity)
            next_order_id += 1

    for record in ordered_depth[1:]:
        for side, levels in (("bid", record.bids), ("ask", record.asks)):
            for price, new_size in levels:
                key = (side, float(price))
                old_size = reference.get(key, 0.0)
                delta = float(new_size) - old_size
                reference[key] = float(new_size)
                if abs(delta) <= 1e-12:
                    continue
                if delta > 0:
                    events.append(market_add(side, price, delta, next_order_id, record.timestamp))
                    next_order_id += 1
                    continue
                decrease = abs(delta)
                aggressor = expected_aggressor_for_decrease(side)
                execute_qty = _consume_trade_quantity(trade_states, price=float(price), aggressor_side=aggressor, timestamp=record.timestamp, needed=decrease, tolerance_ms=tolerance_ms)
                if execute_qty > 1e-12:
                    events.append(BookEvent("EXECUTE", side, float(price), execute_qty, 0, MARKET_ID, record.timestamp))
                cancel_qty = decrease - execute_qty
                if cancel_qty > 1e-12:
                    events.append(BookEvent("CANCEL", side, float(price), cancel_qty, 0, MARKET_ID, record.timestamp))

    consumed = {state.trade.trade_id: float(state.trade.quantity - state.remaining) for state in trade_states}
    return ReconstructionResult(events, consumed)


def _consume_trade_quantity(
    trade_states: list[_TradeState],
    *,
    price: float,
    aggressor_side: str,
    timestamp: int,
    needed: float,
    tolerance_ms: int,
) -> float:
    remaining_need = float(needed)
    consumed = 0.0
    for state in trade_states:
        if remaining_need <= 1e-12:
            break
        trade = state.trade
        if state.remaining <= 1e-12:
            continue
        if trade.price != price or trade.aggressor_side != aggressor_side:
            continue
        if abs(trade.timestamp - timestamp) > tolerance_ms:
            continue
        take = min(remaining_need, state.remaining)
        state.remaining -= take
        remaining_need -= take
        consumed += take
    return consumed

