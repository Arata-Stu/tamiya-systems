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
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.output_scale
        return x.reshape(x.size(0), self.num_points, 2)


class PilotNetBezierTrajectory(nn.Module):
    """
    PilotNet-style camera-to-cubic-Bezier model.
    Input: image (B,C,H,W) or sequence (B,S,C,H,W)
    Output: sampled local trajectory points (B,num_points,2), x-forward/y-left in base_link.
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
            flatten_dim = x.reshape(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 6)

        t = torch.linspace(1.0 / self.num_points, 1.0, self.num_points).view(1, self.num_points, 1)
        self.register_buffer("bezier_t", t)

        self.apply(_init_weights)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def _sample_bezier(self, control_points):
        p0 = torch.zeros(control_points.size(0), 1, 2, device=control_points.device, dtype=control_points.dtype)
        p1 = control_points[:, 0:1, :]
        p2 = control_points[:, 1:2, :]
        p3 = control_points[:, 2:3, :]
        t = self.bezier_t.to(device=control_points.device, dtype=control_points.dtype)
        one_minus_t = 1.0 - t
        return (
            one_minus_t.pow(3) * p0
            + 3.0 * one_minus_t.pow(2) * t * p1
            + 3.0 * one_minus_t * t.pow(2) * p2
            + t.pow(3) * p3
        )

    def forward(self, x):
        if x.dim() == 5:
            x = x[:, -1, :, :, :]

        if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 4 and x.shape[1] == 4:
            x = x[:, :3, :, :]

        x = self.pool(self._forward_conv(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        control_points = torch.tanh(self.fc3(x)).reshape(x.size(0), 3, 2) * self.output_scale
        return self._sample_bezier(control_points)


def infer_trajectory_architecture(state_dict, num_points: int = 20) -> str:
    fc3_weight = state_dict.get("fc3.weight")
    if fc3_weight is None:
        return "direct"
    output_dim = int(fc3_weight.shape[0])
    if output_dim == 6:
        return "bezier"
    if output_dim == int(num_points) * 2:
        return "direct"
    raise ValueError(f"Cannot infer trajectory architecture from fc3.weight shape {tuple(fc3_weight.shape)}")


def create_trajectory_model(
    architecture: str,
    num_points: int = 20,
    input_channels: int = 3,
    input_height: int = 240,
    input_width: int = 320,
    output_scale: float = 10.0,
):
    if architecture == "direct":
        return PilotNetTrajectory(
            num_points=num_points,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            output_scale=output_scale,
        )
    if architecture == "bezier":
        return PilotNetBezierTrajectory(
            num_points=num_points,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            output_scale=output_scale,
        )
    raise ValueError(f"Unknown trajectory architecture: {architecture}")
