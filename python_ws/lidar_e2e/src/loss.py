import torch.nn as nn


class ControlLoss(nn.Module):
    def __init__(self, criterion=nn.SmoothL1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)
