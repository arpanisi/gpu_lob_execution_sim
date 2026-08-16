from __future__ import annotations

from env.episode import make_window_from_events, observations, reset
from env.observation import OBSERVATION_LENGTH
from lob.events import BookEvent


def test_observation_has_locked_fixed_length_and_task_fields() -> None:
    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
        BookEvent("ADD", "bid", 98.0, 1.0, 3, 0, 1001),
    ]
    state = reset(make_window_from_events(events), agent_sides=("buy", "sell"), target_quantity=0.25)
    obs = observations(state)

    assert len(obs) == 2
    assert OBSERVATION_LENGTH == 89
    assert all(len(item) == OBSERVATION_LENGTH for item in obs)
    assert obs[0][80] == 100.0
    assert obs[0][82] == 1.0
    assert obs[0][83] == 0.0
    assert obs[0][80:] == [
        100.0,  # current mid
        0.0,    # elapsed fraction
        1.0,    # remaining target / initial target
        0.0,    # executed quantity / initial target
        100.0,  # arrival mid
        0.0,    # price drift since arrival
        0.0,    # informed-flow proxy
        -1.0,   # resting offset sentinel
        0.0,    # resting quantity sentinel
    ]
