import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from src.model.encoder import LidarEncoder


class TD3Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = LidarEncoder(name="encoder")(obs)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return jnp.tanh(x)


class TD3Critic(nn.Module):
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


def create_td3_states(
    rng,
    obs_shape,
    action_dim,
    actor_lr,
    critic_lr,
):
    rng_actor, rng_c1, rng_c2 = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((1,) + obs_shape)
    dummy_action = jnp.zeros((1, action_dim))

    actor_model = TD3Actor(action_dim=action_dim)
    actor_params = actor_model.init(rng_actor, dummy_obs)
    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(actor_lr),
    )

    critic_model = TD3Critic()
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

    target_actor_params = actor_params
    target_critic1_params = critic1_params
    target_critic2_params = critic2_params

    return (
        actor_state,
        critic1_state,
        critic2_state,
        target_actor_params,
        target_critic1_params,
        target_critic2_params,
    )


@jax.jit
def td3_act_deterministic(actor_state, obs):
    return actor_state.apply_fn(actor_state.params, obs)


@jax.jit
def td3_act(actor_state, obs, rng, exploration_noise):
    action = actor_state.apply_fn(actor_state.params, obs)
    noise = jax.random.normal(rng, shape=action.shape) * exploration_noise
    noisy_action = jnp.clip(action + noise, -1.0, 1.0)
    return noisy_action


@jax.jit
def td3_update_critics(
    actor_state,
    critic1_state,
    critic2_state,
    target_actor_params,
    target_critic1_params,
    target_critic2_params,
    obs,
    actions,
    rewards,
    next_obs,
    terminated,
    rng,
    gamma,
    target_policy_noise,
    target_noise_clip,
):
    next_actions = actor_state.apply_fn(target_actor_params, next_obs)
    noise = jax.random.normal(rng, shape=next_actions.shape) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    target_q1 = critic1_state.apply_fn(target_critic1_params, next_obs, next_actions)
    target_q2 = critic2_state.apply_fn(target_critic2_params, next_obs, next_actions)
    target_q = rewards + gamma * (1.0 - terminated) * jnp.minimum(target_q1, target_q2)
    target_q_sg = jax.lax.stop_gradient(target_q)

    def critic1_loss_fn(params):
        q1 = critic1_state.apply_fn(params, obs, actions)
        return jnp.mean((q1 - target_q_sg) ** 2)

    def critic2_loss_fn(params):
        q2 = critic2_state.apply_fn(params, obs, actions)
        return jnp.mean((q2 - target_q_sg) ** 2)

    critic1_grads = jax.grad(critic1_loss_fn)(critic1_state.params)
    critic2_grads = jax.grad(critic2_loss_fn)(critic2_state.params)
    new_critic1_state = critic1_state.apply_gradients(grads=critic1_grads)
    new_critic2_state = critic2_state.apply_gradients(grads=critic2_grads)

    metrics = {
        "critic1_loss": critic1_loss_fn(new_critic1_state.params),
        "critic2_loss": critic2_loss_fn(new_critic2_state.params),
        "q_target_mean": jnp.mean(target_q),
    }

    return new_critic1_state, new_critic2_state, metrics


@jax.jit
def td3_update_actor_and_targets(
    actor_state,
    critic1_state,
    target_actor_params,
    target_critic1_params,
    obs,
    tau,
):
    def actor_loss_fn(params):
        pi_actions = actor_state.apply_fn(params, obs)
        q_pi = critic1_state.apply_fn(critic1_state.params, obs, pi_actions)
        return -jnp.mean(q_pi)

    actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=actor_grads)

    new_target_actor_params = jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s,
        target_actor_params,
        new_actor_state.params,
    )
    new_target_critic1_params = jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s,
        target_critic1_params,
        critic1_state.params,
    )
    return new_actor_state, new_target_actor_params, new_target_critic1_params, actor_loss


@jax.jit
def td3_soft_update_target_critic2(target_critic2_params, critic2_params, tau):
    return jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s,
        target_critic2_params,
        critic2_params,
    )
