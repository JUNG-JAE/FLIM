import torch
import torch.nn as nn
from conf.global_settings import CHANNEL_SIZE


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(CHANNEL_SIZE, 1, kernel_size=5),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(784, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

