import jax.numpy as jnp
import flax.linen as nn

class LidarEncoder(nn.Module):
    """LiDARデータから特徴量を抽出する1D-CNNバックボーン"""
    
    @nn.compact
    def __call__(self, x):
        # 次元調整 (シーケンスの最後を取得し、チャネル次元を追加)
        if x.ndim == 3:
            x = x[:, -1, :] 
        x = x[..., jnp.newaxis]

        he_init = nn.initializers.he_normal()

        # 1D Conv層 (特徴抽出)
        x = nn.Conv(features=24, kernel_size=(10,), strides=(4,), kernel_init=he_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=36, kernel_size=(8,), strides=(4,), kernel_init=he_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=48, kernel_size=(4,), strides=(2,), kernel_init=he_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,), kernel_init=he_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,), kernel_init=he_init)(x)
        x = nn.relu(x)

        # Flatten (バッチ次元を残して1次元ベクトル化)
        x = x.reshape((x.shape[0], -1))
        
        return x  # 抽出された特徴量ベクトルを返す
