import torch 
import torch.cuda
from torch import nn
from torch.utils import data
from torchinfo import summary

class Conv2dBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size,
            dilation=1, dropout=0.0, pool_size=1,
            activations = nn.Mish
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # padding needed to maintain size

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels, momentum=0.05)
        ]
        if pool_size > 1:
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(activations())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransposeConv2dBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size, dropout=0.0,
            activations = nn.Mish
    ):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels, momentum=0.05)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(activations())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DilatedResConv2dBlock(nn.Module):

    def __init__(
            self, in_channels, mid_channels, out_channels, kernel_size,
            dilation=1, dropout=0.01, activations = nn.Mish
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv2dBlock(in_channels, mid_channels, kernel_size, dilation=dilation),
            Conv2dBlock(mid_channels, out_channels, kernel_size, dropout=dropout)
        )

        self.activation = activations()

    def forward(self, x):
        blocks_output = self.activation(self.blocks(x))
        x = x + blocks_output   # residual connection
        return x
    

class LazyLinearBlock(nn.Module):
    
        def __init__(self, out_features, dropout=0.0, activations = nn.Mish):
            super().__init__()
    
            layers = [
                nn.LazyLinear(out_features),
                nn.BatchNorm1d(out_features, momentum=0.05)
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(activations())
    
            self.block = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.block(x)
        

class LinearBlock(nn.Module):
    
        def __init__(self, in_features, out_features, dropout=0.0, activations = nn.Mish):
            super().__init__()
    
            layers = [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features, momentum=0.05)
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(activations())
    
            self.block = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.block(x)