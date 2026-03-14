import jax.numpy as jnp
import flax.linen as nn
from .encoder import LidarEncoder 
from .head import MLPHead

class Actor(nn.Module):
    """行動を決定するポリシーネットワーク（ONNX出力対象）"""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # 1. エンコーダで特徴抽出
        features = LidarEncoder(name="encoder")(x)
        # 2. MLPで行動を出力
        action = MLPHead(
            output_dim=self.action_dim, 
            last_activation=jnp.tanh, 
            name="actor_mlp"
        )(features)
        return action