
import torch
from torch import nn

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, fin, style_dim=256):
        super().__init__()

        self.norm = nn.InstanceNorm2d(fin, affine=False)
        self.style = nn.Linear(style_dim, fin * 2)

        self.style.bias.data[:fin] = 1
        self.style.bias.data[fin:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out