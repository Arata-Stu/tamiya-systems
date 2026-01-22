import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

### 1. 汎用的な重み初期化関数 ###
def _init_weights(m):
    """
    モジュールに応じた重み初期化を適用する関数
    """
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        # 活性化関数がReLUなので、He初期化 (Kaiming Normal) を使用
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            # バイアスは0で初期化
            init.constant_(m.bias, 0)

### 2. 各モデルクラスに初期化処理を適用 ###

class TinyLidarNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)
        
        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :] 
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
