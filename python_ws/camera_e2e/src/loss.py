import torch.nn as nn


class ControlLoss(nn.Module):
    def __init__(self, criterion=nn.SmoothL1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, targets):
        """
        outputs: (B, 2) または (B, S, 2)
        targets: (B, S, 2)
        """
        if outputs.dim() == 2 and targets.dim() == 3:
            targets = targets[:, -1, :]
        return self.criterion(outputs, targets)

