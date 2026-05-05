import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0)


class PilotNetTrajectory(nn.Module):
    """
    Lightweight PilotNet-style camera-to-path model.
    Input: image (B,C,H,W) or sequence (B,S,C,H,W)
    Output: local trajectory points (B,num_points,2), x-forward/y-left in base_link.
    """

    def __init__(
        self,
        num_points: int = 20,
        input_channels: int = 3,
        input_height: int = 240,
        input_width: int = 320,
        output_scale: float = 10.0,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.output_scale = float(output_scale)

        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            x = self.pool(self._forward_conv(dummy_input))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_points * 2)

        self.apply(_init_weights)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def forward(self, x):
        if x.dim() == 5:
            x = x[:, -1, :, :, :]

        if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 4 and x.shape[1] == 4:
            x = x[:, :3, :, :]

        x = self.pool(self._forward_conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.output_scale
        return x.view(x.size(0), self.num_points, 2)
