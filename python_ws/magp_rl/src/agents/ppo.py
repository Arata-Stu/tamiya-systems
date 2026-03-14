import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from src.model.actor import Actor
from src.model.critic import Critic


def create_train_states(rng, obs_shape, action_dim, actor_lr=3e-4, critic_lr=1e-3):
    """ActorとCriticそれぞれのTrainStateを初期化して返す関数"""
    rng_actor, rng_critic = jax.random.split(rng)
    dummy_obs = jnp.zeros((1,) + obs_shape)

    actor_model = Actor(action_dim=action_dim)
    actor_params = actor_model.init(rng_actor, dummy_obs)
    actor_tx = optax.adam(learning_rate=actor_lr)
    actor_state = TrainState.create(apply_fn=actor_model.apply, params=actor_params, tx=actor_tx)

    critic_model = Critic()
    critic_params = critic_model.init(rng_critic, dummy_obs)
    critic_tx = optax.adam(learning_rate=critic_lr)
    critic_state = TrainState.create(apply_fn=critic_model.apply, params=critic_params, tx=critic_tx)

    return actor_state, critic_state


def calculate_log_prob(mean, log_std, action):
    """正規分布における行動の対数確率(log_prob)を計算する補助関数"""
    std = jnp.exp(log_std)
    var = std**2
    log_scale = log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    return -0.5 * ((action - mean) ** 2) / var - log_scale


@jax.jit
def select_action(actor_state, obs, rng):
    """環境を動かす際に行動をサンプリングする関数"""
    action_mean = actor_state.apply_fn(actor_state.params, obs)

    action_log_std = jnp.zeros_like(action_mean)
    std = jnp.exp(action_log_std)

    noise = jax.random.normal(rng, shape=action_mean.shape)
    action = action_mean + noise * std
    action = jnp.clip(action, -1.0, 1.0)

    log_prob = calculate_log_prob(action_mean, action_log_std, action).sum(axis=-1)
    return action, log_prob


@jax.jit
def select_action_deterministic(actor_state, obs):
    """評価時の決定論的な行動選択"""
    return actor_state.apply_fn(actor_state.params, obs)


@jax.jit
def update_step(
    actor_state,
    critic_state,
    obs,
    actions,
    log_probs_old,
    returns,
    advantages,
    clip_eps=0.2,
    entropy_coef=0.01,
):
    """PPOの1ステップ更新を行う関数"""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def actor_loss_fn(params):
        action_mean = actor_state.apply_fn(params, obs)
        action_log_std = jnp.zeros_like(action_mean)

        log_probs_new = calculate_log_prob(action_mean, action_log_std, actions).sum(axis=-1)
        ratio = jnp.exp(log_probs_new - log_probs_old)

        p1 = ratio * advantages
        p2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -jnp.minimum(p1, p2).mean()

        entropy = (action_log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi)).sum(axis=-1).mean()
        total_actor_loss = actor_loss - entropy_coef * entropy
        return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}

    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (_, actor_metrics), actor_grads = actor_grad_fn(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=actor_grads)

    def critic_loss_fn(params):
        values = critic_state.apply_fn(params, obs).squeeze(-1)
        critic_loss = 0.5 * jnp.mean((returns - values) ** 2)
        return critic_loss, {"critic_loss": critic_loss}

    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    (_, critic_metrics), critic_grads = critic_grad_fn(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=critic_grads)

    metrics = {**actor_metrics, **critic_metrics}
    return new_actor_state, new_critic_state, metrics
