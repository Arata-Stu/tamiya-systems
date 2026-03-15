import jax
import jax.numpy as jnp

class RolloutBuffer:
    """PPO用の経験データを一時保存するバッファ"""
    def __init__(self):
        self.reset()

    def reset(self):
        """新しいエピソード/イテレーションのためにリストを空にする"""
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self, obs, action, reward, done, value, log_prob):
        """1ステップ分のデータを追加"""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get_stacked(self):
        """
        収集したリストをJAXのテンソルに一括変換して返す。
        結果の形状は (num_steps, num_envs, ...) になります。
        """
        return {
            "obs": jnp.stack(self.obs),
            "actions": jnp.stack(self.actions),
            "rewards": jnp.stack(self.rewards),
            "dones": jnp.stack(self.dones),
            "values": jnp.stack(self.values),
            "log_probs": jnp.stack(self.log_probs),
        }

# ==========================================
# JAXネイティブな超高速GAE計算
# ==========================================
@jax.jit
def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    """
    GAE (Generalized Advantage Estimation) を計算する関数。
    jax.lax.scanを使って、未来から過去へ向かって高速に計算します。
    """
    
    def body_fn(carry, step_data):
        gae, next_val = carry
        reward, done, val = step_data
        
        # TD誤差(デルタ)の計算
        # 次のステップが完了(done=1)の場合、未来の価値は0として扱う
        delta = reward + gamma * next_val * (1.0 - done) - val
        
        # Advantageの計算
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        
        return (gae, val), gae

    # データを逆順(時間軸の最後から最初へ)にする
    step_data = (rewards[::-1], dones[::-1], values[::-1])
    
    # jax.lax.scanによる高速なループ計算
    # carryの初期値: (gae=0, next_val=last_value)
    _, advantages = jax.lax.scan(
        body_fn, 
        (jnp.zeros_like(last_value), last_value), 
        step_data
    )
    
    # 結果が逆順で出てくるので、元の時間軸に戻す
    advantages = advantages[::-1]
    
    # 収益(Returns) = Advantage + 状態価値(Value)
    returns = advantages + values
    
    return advantages, returns
