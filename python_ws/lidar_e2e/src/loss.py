import torch
import torch.nn as nn

class ControlLoss(nn.Module):
    def __init__(self, criterion=nn.SmoothL1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, targets):
        """
        outputs: (Batch, OutDim) または (Batch, SeqLen, OutDim)
        targets: (Batch, SeqLen, OutDim) - Datasetが常に時系列で返す想定
        """
        # Case 1: モデルが「最新の1点」のみを出力している場合 (Sequence-to-One)
        # outputs: (B, 2), targets: (B, S, 2)
        if outputs.dim() == 2 and targets.dim() == 3:
            # 正解ラベルの「最新（最後）」のステップだけを抽出して比較
            targets = targets[:, -1, :]
            
        # Case 2: モデルが「時系列全体」を出力している場合 (Sequence-to-Sequence)
        # outputs: (B, S, 2), targets: (B, S, 2)
        # そのまま比較可能
        
        return self.criterion(outputs, targets)