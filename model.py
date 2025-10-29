import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===== 1. UNet模型定义 =====
class TimeEmbedding(nn.Module):
    """时间步嵌入层"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(F.relu(x))
        h = self.norm1(h)
        # 添加时间嵌入
        h = h + self.time_mlp(F.relu(t))[:, :, None, None]
        h = self.conv2(F.relu(h))
        h = self.norm2(h)
        return h + self.shortcut(x)

class UNet(nn.Module):
    """简化的UNet模型"""
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        
        # 编码器
        self.enc1 = ResBlock(in_channels, 64, time_dim)
        self.enc2 = ResBlock(64, 128, time_dim)
        self.enc3 = ResBlock(128, 256, time_dim)
        
        # 瓶颈层
        self.bottleneck = ResBlock(256, 256, time_dim)
        
        # 解码器
        self.dec3 = ResBlock(512, 128, time_dim)
        self.dec2 = ResBlock(256, 64, time_dim)
        self.dec1 = ResBlock(128, 64, time_dim)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        # 时间嵌入
        t = self.time_embed(t)
        
        # 编码器
        e1 = self.enc1(x, t)
        e2 = self.enc2(self.pool(e1), t)
        e3 = self.enc3(self.pool(e2), t)
        
        # 瓶颈
        b = self.bottleneck(self.pool(e3), t)

        b = F.interpolate(b, size=(7, 7), mode='bilinear', align_corners=True)
        
        # 解码器（带跳跃连接）
        d3 = self.dec3(torch.cat([b, e3], dim=1), t)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), t)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), t)
        
        return self.out(d1)