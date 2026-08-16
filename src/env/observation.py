from __future__ import annotations

from lob.constants import EMPTY_PRICE
from lob.matching import OrderBook, mid_price, top_levels
from env.execution import ExecutionTask


OBSERVATION_LENGTH = 89


def agent_resting_order(book: OrderBook, trader_id: int, arrival_mid: float) -> tuple[float, float]:
    for side_arr in (book.bids, book.asks):
        rows = side_arr[(side_arr[:, 0] != EMPTY_PRICE) & (side_arr[:, 3] == float(trader_id))]
        if rows.size:
            row = rows[0]
            return float(row[0] - arrival_mid), float(row[1])
    return -1.0, 0.0


def observation_vector(
    book: OrderBook,
    task: ExecutionTask,
    *,
    trader_id: int,
    current_time: int,
    start_time: int,
    end_time: int,
    informed_flow_value: float = 0.0,
) -> list[float]:
    bid_levels = top_levels(book, "bid", levels=20)
    ask_levels = top_levels(book, "ask", levels=20)
    current_mid = mid_price(book)
    elapsed_denominator = max(1, end_time - start_time)
    elapsed_fraction = min(1.0, max(0.0, (current_time - start_time) / elapsed_denominator))
    resting_offset, resting_quantity = agent_resting_order(book, trader_id, task.arrival_mid)

    values: list[float] = []
    values.extend(float(item) for row in bid_levels for item in row)
    values.extend(float(item) for row in ask_levels for item in row)
    values.extend(
        [
            current_mid,
            elapsed_fraction,
            task.remaining_target / task.initial_target,
            task.executed_quantity / task.initial_target,
            task.arrival_mid,
            current_mid - task.arrival_mid,
            float(informed_flow_value),
            resting_offset,
            resting_quantity,
        ]
    )
    if len(values) != OBSERVATION_LENGTH:
        raise RuntimeError(f"observation length {len(values)} != {OBSERVATION_LENGTH}")
    return values
