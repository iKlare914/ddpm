import torch as th
from torch.nn import functional as F
from torch import nn
import tqdm.auto as tqdm
from abc import abstractmethod

def zero_init(module: nn.Module):
    """
    Initialize all parameters of a module with zero
    """
    for p in module.parameters:
        p = p.detach().zero_()
    return module

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1/self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x

class TimeEmbeddedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply time embedding to input x
        """

class ResidualBlock(TimeEmbeddedBlock):
    def __init__(self, in_channel, out_channel, emb_channel, dropout, is_upsample):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            nn.SiLU()
        )
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1)
        if is_upsample:
            self.x_upd = UpSampleBlock(out_channel, out_channel, 2)
            self.h_upd = UpSampleBlock(out_channel, out_channel, 2)
        else:
            self.x_upd = DownSampleBlock(out_channel, out_channel, 2)
            self.h_upd = DownSampleBlock(out_channel, out_channel, 2)

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_init(nn.Conv2d(out_channel, out_channel, 3, 1))
        )

        # get independent representation of time embedding for every block
        if emb_channel != out_channel:
            raise ValueError(f"Channel of time embedding must be identical to out channel, but got embedding channel: {emb_channel}, out channel: {out_channel}")
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channel, emb_channel)
        )

    def forward(self, x: th.Tensor, emb: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
            emb: Tensor [N, C]
        """
        h = self.in_layers(x)
        h = self.h_upd(h)
        h = self.conv(h)
        x = self.x_upd(x)

        emb = self.emb_layer(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)

        h = self.out_layers(emb + h)
        return x + h

class TimeSequentialBlock(nn.Sequential, TimeEmbeddedBlock):
    """
    A sequential module that dispatch appropriate input for modules
    """
    def forward(self, x, emb):
        """:w

        Args:
            x: Tensor [B, C, H, W]
            emb: Tensor [N, C]
        """

        for layer in self:
            if isinstance(layer, TimeEmbeddedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x