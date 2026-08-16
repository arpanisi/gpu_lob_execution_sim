from __future__ import annotations

import importlib.util

import pytest

from agents.jax_ippo import ACTION_DIM, HIDDEN_DIM, require_ippo_jax


def test_ippo_network_shape_constants_are_locked_small_mlp_choice() -> None:
    assert ACTION_DIM == 2
    assert HIDDEN_DIM == 128


def test_ippo_jax_guard_is_explicit_when_jax_is_missing() -> None:
    if importlib.util.find_spec("jax") is not None:
        require_ippo_jax()
    else:
        with pytest.raises(ModuleNotFoundError):
            require_ippo_jax()
