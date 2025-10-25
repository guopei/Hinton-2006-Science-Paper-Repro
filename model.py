import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.time_embedding = nn.Linear(1, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        

    def forward(self, x, t):

        time_emb = self.time_embedding(t)
        
        h = self.conv(h)

        h = h + time_emb[:, :, None, None]
        h = self.norm(h)
        h = self.activation(h)
        
        return h

class UNet(nn.Module):
    """Simplified U-Net for MNIST autoencoder"""
    def __init__(self, hidden_dim=256, time_emb_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Input projection (784 -> 28x28x1)
        self.input_proj = nn.Linear(input_dim + 1, 28 * 28)  # +1 for time
        
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
        self.final_proj = nn.Linear(28 * 28, input_dim)
        
        # Downsampling and upsampling
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t):
        batch_size = x.shape[0]
        
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Concatenate input with time and project to image space
        x_with_time = torch.cat([x, t], dim=-1)
        x_img = self.input_proj(x_with_time).view(batch_size, 1, 28, 28)
        
        # Encoder
        e1 = self.enc1(x_img, time_emb)
        e2 = self.enc2(self.downsample(e1), time_emb)
        e3 = self.enc3(self.downsample(e2), time_emb)
        
        # Bottleneck
        b = self.bottleneck(self.downsample(e3), time_emb)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1), time_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), time_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), time_emb)
        
        # Output
        out_img = self.output_proj(d1)
        out = self.final_proj(out_img.view(batch_size, -1))
        
        return out

    def step(self, x, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x.shape[0], 1)
        # Use simple Euler method for stability
        return x + (t_end - t_start) * self(x, t_start)