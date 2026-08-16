from __future__ import annotations

import pytest

from data.date_ranges import iter_dates


def test_iter_dates_inclusive() -> None:
    assert iter_dates("2024-01-01", "2024-01-03") == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_iter_dates_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        iter_dates("2024-01-03", "2024-01-01")
