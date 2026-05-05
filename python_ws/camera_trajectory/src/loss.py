import torch
import torch.nn as nn


class TrajectoryLoss(nn.Module):
    def __init__(self, criterion=nn.SmoothL1Loss(), heading_weight: float = 0.2):
        super().__init__()
        self.criterion = criterion
        self.heading_weight = float(heading_weight)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        outputs: (B, P, 2)
        targets: (B, P, 2)
        """
        point_loss = self.criterion(outputs, targets)
        if self.heading_weight <= 0.0 or outputs.shape[1] < 2:
            return point_loss

        output_delta = outputs[:, 1:, :] - outputs[:, :-1, :]
        target_delta = targets[:, 1:, :] - targets[:, :-1, :]
        heading_loss = self.criterion(output_delta, target_delta)
        return point_loss + self.heading_weight * heading_loss
