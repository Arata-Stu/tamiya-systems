import flax.linen as nn
from .encoder import LidarEncoder
from .head import MLPHead


class Critic(nn.Module):
    """状態の価値を評価するネットワーク"""
    
    @nn.compact
    def __call__(self, x):
        # 1. Critic用のエンコーダで特徴抽出
        features = LidarEncoder(name="encoder")(x)
        # 2. MLPで価値（スカラー値）を出力
        value = MLPHead(
            output_dim=1, 
            last_activation=None, 
            name="critic_mlp"
        )(features)
        return value