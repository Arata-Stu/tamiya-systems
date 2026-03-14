import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

from src.model.encoder import LidarEncoder


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SACActor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = LidarEncoder(name="encoder")(obs)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class SACCritic(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        x = LidarEncoder(name="encoder")(obs)
        x = jnp.concatenate([x, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


def create_sac_states(
    rng,
    obs_shape,
    action_dim,
    actor_lr,
    critic_lr,
    alpha_lr,
    init_temperature,
):
    rng_actor, rng_c1, rng_c2 = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((1,) + obs_shape)
    dummy_action = jnp.zeros((1, action_dim))

    actor_model = SACActor(action_dim=action_dim)
    actor_params = actor_model.init(rng_actor, dummy_obs)
    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(actor_lr),
    )

    critic_model = SACCritic()
    critic1_params = critic_model.init(rng_c1, dummy_obs, dummy_action)
    critic2_params = critic_model.init(rng_c2, dummy_obs, dummy_action)
    critic1_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic1_params,
        tx=optax.adam(critic_lr),
    )
    critic2_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic2_params,
        tx=optax.adam(critic_lr),
    )

    alpha_state = TrainState.create(
        apply_fn=lambda params: params["log_alpha"],
        params={"log_alpha": jnp.array(jnp.log(init_temperature), dtype=jnp.float32)},
        tx=optax.adam(alpha_lr),
    )

    target_critic1_params = critic1_params
    target_critic2_params = critic2_params

    return actor_state, critic1_state, critic2_state, target_critic1_params, target_critic2_params, alpha_state


def _gaussian_log_prob(noise, log_std):
    return -0.5 * (noise**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))


def _sample_action_and_log_prob(actor_apply_fn, actor_params, obs, rng):
    mean, log_std = actor_apply_fn(actor_params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, shape=mean.shape)
    pre_tanh = mean + std * noise
    action = jnp.tanh(pre_tanh)

    log_prob = _gaussian_log_prob(noise, log_std).sum(axis=-1)
    correction = jnp.log(1.0 - action**2 + 1e-6).sum(axis=-1)
    log_prob = log_prob - correction
    return action, log_prob


@jax.jit
def sac_act(actor_state, obs, rng):
    action, _ = _sample_action_and_log_prob(actor_state.apply_fn, actor_state.params, obs, rng)
    return action


@jax.jit
def sac_act_deterministic(actor_state, obs):
    mean, _ = actor_state.apply_fn(actor_state.params, obs)
    return jnp.tanh(mean)


@jax.jit
def sac_update_step(
    actor_state,
    critic1_state,
    critic2_state,
    target_critic1_params,
    target_critic2_params,
    alpha_state,
    obs,
    actions,
    rewards,
    next_obs,
    terminated,
    rng,
    gamma,
    tau,
    target_entropy,
):
    rng_next, rng_actor = jax.random.split(rng)

    alpha = jnp.exp(alpha_state.params["log_alpha"])

    next_actions, next_logp = _sample_action_and_log_prob(
        actor_state.apply_fn, actor_state.params, next_obs, rng_next
    )
    next_q1 = critic1_state.apply_fn(target_critic1_params, next_obs, next_actions)
    next_q2 = critic2_state.apply_fn(target_critic2_params, next_obs, next_actions)
    next_v = jnp.minimum(next_q1, next_q2) - alpha * next_logp
    target_q = rewards + gamma * (1.0 - terminated) * next_v

    def critic1_loss_fn(params):
        q1 = critic1_state.apply_fn(params, obs, actions)
        loss = jnp.mean((q1 - jax.lax.stop_gradient(target_q)) ** 2)
        return loss

    def critic2_loss_fn(params):
        q2 = critic2_state.apply_fn(params, obs, actions)
        loss = jnp.mean((q2 - jax.lax.stop_gradient(target_q)) ** 2)
        return loss

    critic1_grads = jax.grad(critic1_loss_fn)(critic1_state.params)
    critic2_grads = jax.grad(critic2_loss_fn)(critic2_state.params)
    new_critic1_state = critic1_state.apply_gradients(grads=critic1_grads)
    new_critic2_state = critic2_state.apply_gradients(grads=critic2_grads)

    def actor_loss_fn(params):
        sampled_actions, logp = _sample_action_and_log_prob(actor_state.apply_fn, params, obs, rng_actor)
        q1_pi = new_critic1_state.apply_fn(new_critic1_state.params, obs, sampled_actions)
        q2_pi = new_critic2_state.apply_fn(new_critic2_state.params, obs, sampled_actions)
        q_pi = jnp.minimum(q1_pi, q2_pi)
        loss = jnp.mean(alpha * logp - q_pi)
        return loss, logp

    (actor_loss, logp_pi), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=actor_grads)

    def alpha_loss_fn(params):
        log_alpha = params["log_alpha"]
        loss = -jnp.mean(log_alpha * jax.lax.stop_gradient(logp_pi + target_entropy))
        return loss

    alpha_grads = jax.grad(alpha_loss_fn)(alpha_state.params)
    new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

    new_target_critic1_params = jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s,
        target_critic1_params,
        new_critic1_state.params,
    )
    new_target_critic2_params = jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s,
        target_critic2_params,
        new_critic2_state.params,
    )

    critic1_loss = critic1_loss_fn(new_critic1_state.params)
    critic2_loss = critic2_loss_fn(new_critic2_state.params)
    alpha_loss = alpha_loss_fn(new_alpha_state.params)

    metrics = {
        "actor_loss": actor_loss,
        "critic1_loss": critic1_loss,
        "critic2_loss": critic2_loss,
        "alpha_loss": alpha_loss,
        "alpha": jnp.exp(new_alpha_state.params["log_alpha"]),
        "q_target_mean": jnp.mean(target_q),
    }

    return (
        new_actor_state,
        new_critic1_state,
        new_critic2_state,
        new_target_critic1_params,
        new_target_critic2_params,
        new_alpha_state,
        metrics,
    )
