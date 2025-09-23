import torch
import torch.nn as nn
import torch.nn.functional as F

from .hex_conv_2d import HexConv2d
from nonaga.hexboard import HexBoard
from nonaga.encoder import Encoder
from nonaga.actions import MoveAction, UpAction, DownAction


class AlphaZeroModel(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            HexConv2d(3, num_hidden),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.moveActionPolicyHead = nn.Sequential(
            HexConv2d(num_hidden, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * Encoder.SIZE * Encoder.SIZE, Encoder.MOVE_ACTION_SIZE)
        )
        self.upActionPolicyHead = nn.Sequential(
            HexConv2d(num_hidden, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * Encoder.SIZE * Encoder.SIZE, Encoder.HEX_ACTION_SIZE)
        )

        self.downActionPolicyHead = nn.Sequential(
            HexConv2d(num_hidden, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * Encoder.SIZE * Encoder.SIZE, Encoder.HEX_ACTION_SIZE)
        )
        
        self.moveActionValueHead = ValueHead(num_hidden)
        self.upActionValueHead = ValueHead(num_hidden)
        self.downActionValueHead = ValueHead(num_hidden)
        
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        moveActionPolicy = self.moveActionPolicyHead(x)
        upActionPolicy = self.upActionPolicyHead(x)
        downActionPolicy = self.downActionPolicyHead(x)
        moveActionValue = self.moveActionValueHead(torch.cat([x], dim=1))
        upActionValue = self.upActionValueHead(torch.cat([x], dim=1))
        downActionValue = self.downActionValueHead(torch.cat([x], dim=1))
        return moveActionPolicy, upActionPolicy, downActionPolicy, moveActionValue, upActionValue, downActionValue
    
class ValueHead(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv = nn.Conv2d(num_hidden, 1, kernel_size=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(Encoder.SIZE * Encoder.SIZE, num_hidden)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = HexConv2d(num_hidden, num_hidden)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = HexConv2d(num_hidden, num_hidden)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x