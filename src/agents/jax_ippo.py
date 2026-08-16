from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.ippo_config import LOCKED_IPPO_CONFIG, PPOConfig
from env.observation import OBSERVATION_LENGTH
from lob.jax_matching import require_jax

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    jax = None
    jnp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


ACTION_DIM = 2
HIDDEN_DIM = 128


@dataclass(frozen=True)
class AdamState:
    m: tuple
    v: tuple
    step: int


def require_ippo_jax() -> None:
    require_jax()
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("JAX is required for agents.jax_ippo") from _IMPORT_ERROR


if jnp is not None:

    def init_agent_params(key, obs_dim: int = OBSERVATION_LENGTH, hidden_dim: int = HIDDEN_DIM):
        keys = jax.random.split(key, 7)
        return (
            _dense_init(keys[0], obs_dim, hidden_dim),
            _dense_init(keys[1], hidden_dim, hidden_dim),
            _dense_init(keys[2], hidden_dim, ACTION_DIM),
            _dense_init(keys[3], hidden_dim, 1),
            jnp.array([-1.0, -1.0], dtype=jnp.float32),
        )


    def init_two_agent_params(seed: int = 0):
        keys = jax.random.split(jax.random.PRNGKey(seed), LOCKED_IPPO_CONFIG.agents_per_environment)
        return tuple(init_agent_params(key) for key in keys)


    def init_adam(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(zeros, zeros, 0)


    def policy_value(params, obs):
        layer1, layer2, policy_head, value_head, log_std = params
        x = _tanh_dense(obs, layer1)
        x = _tanh_dense(x, layer2)
        mean = _linear(x, policy_head)
        value = _linear(x, value_head).squeeze(-1)
        return mean, log_std, value


    def sample_action(params, obs, key):
        mean, log_std, value = policy_value(params, obs)
        std = jnp.exp(log_std)
        raw = mean + std * jax.random.normal(key, mean.shape)
        clipped = jnp.stack(
            [
                jnp.clip(raw[:, 0], -5.0, 10.0),
                jnp.maximum(raw[:, 1], 0.0),
            ],
            axis=1,
        )
        log_prob = gaussian_log_prob(raw, mean, log_std)
        return clipped, log_prob, value


    def gaussian_log_prob(action, mean, log_std):
        var = jnp.exp(2.0 * log_std)
        per_dim = -0.5 * (((action - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        return jnp.sum(per_dim, axis=-1)


    def ppo_update(params, adam_state: AdamState, batch: dict[str, jnp.ndarray], config: PPOConfig = LOCKED_IPPO_CONFIG):
        def loss_fn(p):
            mean, log_std, values = policy_value(p, batch["obs"])
            log_probs = gaussian_log_prob(batch["actions"], mean, log_std)
            ratio = jnp.exp(log_probs - batch["old_log_probs"])
            clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            policy_loss = -jnp.mean(jnp.minimum(ratio * batch["advantages"], clipped_ratio * batch["advantages"]))
            value_loss = 0.5 * jnp.mean((values - batch["returns"]) ** 2)
            entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
            return policy_loss + value_loss - config.entropy_coefficient * entropy

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updated_params, updated_adam = adam_step(params, grads, adam_state, config.learning_rate)
        return updated_params, updated_adam, loss


    def adam_step(params, grads, state: AdamState, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        step = state.step + 1
        m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state.m, grads)
        v = jax.tree_util.tree_map(lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), state.v, grads)
        m_hat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta1**step), m)
        v_hat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta2**step), v)
        updated = jax.tree_util.tree_map(lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
        return updated, AdamState(m, v, step)


    def action_summary(params, observations: np.ndarray) -> dict[str, float]:
        obs = jnp.asarray(observations, dtype=jnp.float32)
        mean, log_std, _ = policy_value(params, obs)
        mean_np = np.asarray(mean)
        std_np = np.asarray(jnp.exp(log_std))
        return {
            "offset_mean": float(mean_np[:, 0].mean()),
            "offset_std": float(mean_np[:, 0].std()),
            "quantity_mean": float(mean_np[:, 1].mean()),
            "quantity_std": float(mean_np[:, 1].std()),
            "policy_std_offset": float(std_np[0]),
            "policy_std_quantity": float(std_np[1]),
        }


    def _dense_init(key, in_dim: int, out_dim: int):
        weight_key, _ = jax.random.split(key)
        scale = jnp.sqrt(2.0 / float(in_dim))
        weight = jax.random.normal(weight_key, (in_dim, out_dim), dtype=jnp.float32) * scale
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        return weight, bias


    def _linear(x, layer):
        weight, bias = layer
        return x @ weight + bias


    def _tanh_dense(x, layer):
        return jnp.tanh(_linear(x, layer))
else:

    def init_two_agent_params(seed: int = 0):
        require_ippo_jax()


    def init_adam(params):
        require_ippo_jax()


    def policy_value(params, obs):
        require_ippo_jax()


    def sample_action(params, obs, key):
        require_ippo_jax()


    def gaussian_log_prob(action, mean, log_std):
        require_ippo_jax()


    def ppo_update(params, adam_state, batch, config=LOCKED_IPPO_CONFIG):
        require_ippo_jax()


    def adam_step(params, grads, state, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        require_ippo_jax()


    def action_summary(params, observations: np.ndarray):
        require_ippo_jax()
