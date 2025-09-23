import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HexConv2d(nn.Module):
    DIRS = [(0,0), (+1,0), (0,+1), (-1,+1), (-1,0), (0,-1), (+1,-1)]
    
    def __init__(self, in_channels, out_channels, bias=True, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 7, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(out_channels, **factory_kwargs)) if bias else None

        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=math.sqrt(5))
        if bias:
            fan_in = in_channels * 7
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _shift_map(x, dy, dx):
        sx, sy = -dx, -dy
        y = torch.roll(x, shifts=(sy, sx), dims=(2, 3))
        if sy > 0: y[:, :, :sy, :] = 0
        elif sy < 0: y[:, :, sy:, :] = 0
        if sx > 0: y[:, :, :, :sx] = 0
        elif sx < 0: y[:, :, :, sx:] = 0
        return y
    
    @staticmethod
    def _get_mask(x):
        _, _, H, W = x.shape
        assert H == W and H % 2 == 1
        size = H // 2
        q = torch.arange(-size, size+1, device=x.device)
        r = torch.arange(-size, size+1, device=x.device)
        Q, R = torch.meshgrid(q, r, indexing='ij')
        S = -Q - R
        mask = ((Q.abs() <= size) & (R.abs() <= size) & (S.abs() <= size)).to(dtype=x.dtype)
        return mask.view(1,1,H,W)
    
    def forward(self, x):
        mask = self._get_mask(x)
        x = x * mask
        neighbor_maps = [
            (self._shift_map(x, *dir) * mask) for dir in self.DIRS
        ]
        stacked_neighbors = torch.cat(neighbor_maps, dim=1)
        w = self.weight.view(self.weight.shape[0], -1, 1, 1)
        y = F.conv2d(stacked_neighbors, w, self.bias, stride=1, padding=0)
        y = y * mask
        return y

