import torch
import torch.nn as nn
import numpy as np

class Colorizer(nn.Module):
    def __init__(self, input_size=128):
        # TODO: figure out how big our inputs are
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x