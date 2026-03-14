import jax
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.envs.f110_wrapper import F110EnvWrapper
from src.agents.ppo import create_train_states, select_action, update_step
from src.utils.buffer import RolloutBuffer, compute_gae

# ★ シミュレータ本体のインポート
from f110_jax.simulator import F110JaxSimulator, Integrator

def generate_initial_poses(num_agents):
    """車両同士の重なりを避けて初期姿勢を生成"""
    poses = np.zeros((num_agents, 3), dtype=np.float32)
    poses[:, 0] = 0.0
    poses[:, 1] = np.linspace(0.0, 0.4 * max(num_agents - 1, 0), num_agents)
    poses[:, 2] = 0.0
    return poses

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    print("=== JAX F1TENTH PPO Training Start ===")
    
    rng = jax.random.PRNGKey(cfg.train.seed)
    rng, rng_agent = jax.random.split(rng, 2)

    # ★ 本物のシミュレータを初期化
    sim = F110JaxSimulator(
        map_path=cfg.env.map_path,
        map_ext=cfg.env.map_ext,
        num_agents=cfg.train.num_agents,
        integrator=Integrator.RK4
    )
    env = F110EnvWrapper(sim, cfg.env)
    
    obs_shape = (cfg.env.obs_dim,)
    actor_state, critic_state = create_train_states(
        rng_agent, obs_shape, cfg.env.action_dim, 
        actor_lr=cfg.train.actor_lr, critic_lr=cfg.train.critic_lr
    )

    buffer = RolloutBuffer()
    num_updates = cfg.train.total_timesteps // (cfg.train.num_steps * cfg.train.num_agents)
    
    # ★ 環境の初回リセット
    poses = generate_initial_poses(cfg.train.num_agents)
    obs = env.reset(poses)

    for update in range(num_updates):
        buffer.reset()
        
        for step in tqdm(range(cfg.train.num_steps), desc=f"Update {update+1}/{num_updates}"):
            rng, rng_action = jax.random.split(rng)
            
            action, log_prob = select_action(actor_state, obs, rng_action)
            value = critic_state.apply_fn(critic_state.params, obs).squeeze()
            
            # ★ 環境のステップ実行 (stateは不要)
            next_obs, reward, done, info = env.step(action)
            
            buffer.add(obs, action, reward, done, value, log_prob)
            obs = next_obs
            
        # --- GAE計算とネットワーク更新 ---
        last_value = critic_state.apply_fn(critic_state.params, obs).squeeze()
        data = buffer.get_stacked()
        advantages, returns = compute_gae(
            data["rewards"], data["values"], data["dones"], last_value,
            gamma=cfg.train.gamma, gae_lambda=cfg.train.gae_lambda
        )

        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        b_obs = flatten(data["obs"])
        b_actions = flatten(data["actions"])
        b_log_probs = flatten(data["log_probs"])
        b_returns = flatten(returns)
        b_advantages = flatten(advantages)

        for epoch in range(cfg.train.update_epochs):
            actor_state, critic_state, metrics = update_step(
                actor_state, critic_state,
                b_obs, b_actions, b_log_probs, b_returns, b_advantages,
                clip_eps=cfg.train.clip_eps,
                entropy_coef=cfg.train.entropy_coef
            )
            
        print(f"Update {update+1} | Actor Loss: {metrics['actor_loss']:.4f} | Critic Loss: {metrics['critic_loss']:.4f}")

if __name__ == "__main__":
    main()