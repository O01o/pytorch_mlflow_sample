import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 出力サイズ: 32x30x30
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 出力サイズ: 64x28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 入力サイズに合わせて修正
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # (バッチサイズ, 64 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x