import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0)


class PilotNetControl(nn.Module):
    """
    NVIDIA PilotNetベースのエンドツーエンド制御モデル。
    入力: 画像 (B, C, H, W) もしくは時系列 (B, S, C, H, W)
    出力: [steer, speed]
    """

    def __init__(
        self,
        num_outputs: int = 2,
        input_channels: int = 3,
        input_height: int = 66,
        input_width: int = 200,
        pooled_height: int = 1,
        pooled_width: int = 18,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pooled_height, pooled_width))

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            x = self.adaptive_pool(self._forward_conv(dummy_input))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, num_outputs)

        self.apply(_init_weights)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def forward(self, x):
        # Sequence-to-One
        if x.dim() == 5:
            x = x[:, -1, :, :, :]

        # NHWC -> NCHW
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 4 and x.shape[1] == 4:
            x = x[:, :3, :, :]

        x = self._forward_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x
