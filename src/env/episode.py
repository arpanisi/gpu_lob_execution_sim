from __future__ import annotations

from dataclasses import dataclass

from data.informed_flow import FlowSignal, compute_informed_flow_signals, current_flow_value
from env.execution import AgentAction, ExecutionTask, execute_action, force_terminal_fill, initial_task
from env.observation import observation_vector
from lob.constants import AGENT_ID_START, EPISODE_LENGTH_MS, STEP_EVENT_COUNT
from lob.events import BookEvent
from lob.matching import Fill, OrderBook, apply_event, cancel_trader_orders, clone_book, mid_price


@dataclass(frozen=True)
class EpisodeWindow:
    start_ts: int
    end_ts: int
    initial_book: OrderBook
    events: tuple[BookEvent, ...]
    flow_signals: dict[int, FlowSignal] | None = None
    flat_event_offset: int = 0


@dataclass
class EpisodeState:
    window: EpisodeWindow
    book: OrderBook
    position: int
    current_time: int
    tasks: list[ExecutionTask]
    done: bool = False


@dataclass(frozen=True)
class StepResult:
    rewards: tuple[float, ...]
    fills: tuple[Fill, ...]
    done: bool
    processed_events: int


def make_window_from_events(
    events: list[BookEvent],
    *,
    episode_length_ms: int = EPISODE_LENGTH_MS,
    flat_event_offset: int = 0,
) -> EpisodeWindow:
    if not events:
        raise ValueError("cannot build an episode window from an empty event list")
    ordered = tuple(sorted(events, key=lambda event: event.timestamp))
    start_ts = ordered[0].timestamp
    end_ts = min(start_ts + episode_length_ms, ordered[-1].timestamp)
    book = OrderBook.empty()
    position = 0
    while position < len(ordered) and ordered[position].timestamp == start_ts:
        apply_event(book, ordered[position], validation=False)
        position += 1
    return EpisodeWindow(
        start_ts,
        end_ts,
        clone_book(book),
        ordered[position:],
        compute_informed_flow_signals(list(ordered[position:])),
        flat_event_offset + position,
    )


def reset(window: EpisodeWindow, *, agent_sides: tuple[str, ...] = ("buy",), target_quantity: float = 0.05) -> EpisodeState:
    book = clone_book(window.initial_book)
    arrival = mid_price(book)
    tasks = [initial_task(side, target_quantity, arrival) for side in agent_sides]  # type: ignore[arg-type]
    return EpisodeState(window, book, position=0, current_time=window.start_ts, tasks=tasks)


def step(state: EpisodeState, actions: list[AgentAction], *, k_events: int = STEP_EVENT_COUNT) -> StepResult:
    if state.done:
        return StepResult(tuple(0.0 for _ in state.tasks), (), True, 0)
    if len(actions) != len(state.tasks):
        raise ValueError("one action is required for each active agent")
    action_ts = state.current_time + 1
    all_fills: list[Fill] = []
    rewards = [0.0 for _ in state.tasks]
    for idx, action in enumerate(actions):
        trader_id = AGENT_ID_START + idx
        cancel_trader_orders(state.book, trader_id)
        task, reward, fills = execute_action(state.book, state.tasks[idx], action, trader_id=trader_id, timestamp=action_ts)
        state.tasks[idx] = task
        rewards[idx] += reward
        all_fills.extend(fills)

    end_pos = state.position
    historical: list[BookEvent] = []
    while end_pos < len(state.window.events) and len(historical) < k_events:
        event = state.window.events[end_pos]
        if event.timestamp > state.window.end_ts:
            break
        historical.append(event)
        end_pos += 1
    for event in sorted(historical, key=lambda item: item.timestamp):
        all_fills.extend(apply_event(state.book, event, validation=False))
        state.current_time = event.timestamp
    state.position = end_pos

    no_more_window_events = state.position >= len(state.window.events) or (
        state.position < len(state.window.events) and state.window.events[state.position].timestamp > state.window.end_ts
    )
    if no_more_window_events or state.current_time >= state.window.end_ts:
        state.done = True
        terminal_ts = max(state.current_time, state.window.end_ts) + 1
        for idx, task in enumerate(state.tasks):
            trader_id = AGENT_ID_START + idx
            cancel_trader_orders(state.book, trader_id)
            task, reward, fills = force_terminal_fill(state.book, task, trader_id=trader_id, timestamp=terminal_ts)
            state.tasks[idx] = task
            rewards[idx] += reward
            all_fills.extend(fills)
    return StepResult(tuple(rewards), tuple(all_fills), state.done, len(historical))


def observations(state: EpisodeState) -> tuple[list[float], ...]:
    flow_value = 0.0
    if state.window.flow_signals is not None:
        flow_value = current_flow_value(state.window.flow_signals, state.position)
    return tuple(
        observation_vector(
            state.book,
            task,
            trader_id=AGENT_ID_START + idx,
            current_time=state.current_time,
            start_time=state.window.start_ts,
            end_time=state.window.end_ts,
            informed_flow_value=flow_value,
        )
        for idx, task in enumerate(state.tasks)
    )
