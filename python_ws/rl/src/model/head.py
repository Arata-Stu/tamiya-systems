import flax.linen as nn
from typing import Callable, Optional

class MLPHead(nn.Module):
    """ActorやCriticで再利用可能なMLP"""
    output_dim: int
    last_activation: Optional[Callable] = None  # 最後の層の活性化関数 (tanh等)

    @nn.compact
    def __call__(self, x):
        he_init = nn.initializers.he_normal()
        
        x = nn.Dense(features=100, kernel_init=he_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=50, kernel_init=he_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10, kernel_init=he_init)(x)
        x = nn.relu(x)
        
        # 出力層
        x = nn.Dense(features=self.output_dim, kernel_init=he_init)(x)
        
        # 必要に応じて最後の活性化関数を適用
        if self.last_activation is not None:
            x = self.last_activation(x)
            
        return x