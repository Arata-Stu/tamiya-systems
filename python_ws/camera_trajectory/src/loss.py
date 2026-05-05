import torch
import torch.nn as nn


class TrajectoryLoss(nn.Module):
    def __init__(
        self,
        criterion=nn.SmoothL1Loss(),
        heading_weight: float = 0.2,
        curvature_weight: float = 0.5,
        smoothness_weight: float = 0.05,
        progress_weight: float = 0.05,
    ):
        super().__init__()
        self.criterion = criterion
        self.heading_weight = float(heading_weight)
        self.curvature_weight = float(curvature_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.progress_weight = float(progress_weight)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        outputs: (B, P, 2)
        targets: (B, P, 2)
        """
        loss = self.criterion(outputs, targets)

        if outputs.shape[1] >= 2:
            output_delta = outputs[:, 1:, :] - outputs[:, :-1, :]
            target_delta = targets[:, 1:, :] - targets[:, :-1, :]

            if self.heading_weight > 0.0:
                heading_loss = self.criterion(output_delta, target_delta)
                loss = loss + self.heading_weight * heading_loss

            if self.progress_weight > 0.0:
                target_dx = target_delta[..., 0]
                output_dx = output_delta[..., 0]
                forward_mask = target_dx > 0.0
                if forward_mask.any():
                    progress_loss = torch.relu(-output_dx[forward_mask]).mean()
                    loss = loss + self.progress_weight * progress_loss

        if outputs.shape[1] >= 3:
            output_curvature = outputs[:, 2:, :] - 2.0 * outputs[:, 1:-1, :] + outputs[:, :-2, :]
            target_curvature = targets[:, 2:, :] - 2.0 * targets[:, 1:-1, :] + targets[:, :-2, :]

            if self.curvature_weight > 0.0:
                curvature_loss = self.criterion(output_curvature, target_curvature)
                loss = loss + self.curvature_weight * curvature_loss

            if self.smoothness_weight > 0.0:
                smoothness_loss = torch.mean(output_curvature.pow(2))
                loss = loss + self.smoothness_weight * smoothness_loss

        return loss
