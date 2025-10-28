import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.module import T


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.squeeze()
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
           self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
           self.skip = nn.Identity()


    def forward(self, x, t):

        time_proj = self.time_proj(t)
        
        h = self.conv(x)

        h = h + time_proj[:, :, None, None]
        h = self.norm(h)
        h = self.activation(h)

        x = self.skip(x)

        x += h
        
        return x

class UNet(nn.Module):
    """Simplified U-Net for MNIST autoencoder"""
    def __init__(self, time_emb_dim=128):
        super().__init__()

        self.time_embedding = TimeEmbedding(dim=128)
        
        # Encoder
        self.enc1 = ResidualBlock(1, 64, time_emb_dim)
        self.enc2 = ResidualBlock(64, 128, time_emb_dim)
        self.enc3 = ResidualBlock(128, 256, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512, time_emb_dim)
        
        # Decoder
        self.dec3 = ResidualBlock(512 + 256, 256, time_emb_dim)
        self.dec2 = ResidualBlock(256 + 128, 128, time_emb_dim)
        self.dec1 = ResidualBlock(128 + 64, 64, time_emb_dim)
        
        # Output projection
        self.output_proj = nn.Conv2d(64, 1, 1)
        
        # Downsampling and upsampling
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t):
        batch_size = x.shape[0]
        
        # Encoder
        e1 = self.enc1(x, self.time_embedding(t))
        e2 = self.enc2(self.downsample(e1), self.time_embedding(t))
        e3 = self.enc3(self.downsample(e2), self.time_embedding(t))
        
        # Bottleneck
        b = self.bottleneck(e3, self.time_embedding(t))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([b, e3], dim=1), self.time_embedding(t))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), self.time_embedding(t))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), self.time_embedding(t))
        
        # Output
        out = self.output_proj(d1)
        
        return out

    def step(self, x, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x.shape[0], 1)
        # Use simple Euler method for stability
        return x + (t_end[..., None, None] - t_start[..., None, None]) * self(x, t_start)



if __name__ == "__main__":
    model = UNet()
    print(model)

    inputs = torch.randn(10, 1, 28, 28)
    t = torch.rand(10, 1)
    output = model(inputs, t)
    assert output.shape == inputs.shape