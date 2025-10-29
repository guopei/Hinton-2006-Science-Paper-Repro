import os
import time
import math

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import PILImage
from PIL import Image 
import torch.nn.functional as F

from model import UNet
from data import create_mnist_dataloaders, visualize_mnist_data

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def main():
    print("Hello from autoencoder!")
    device="cuda"

    train_epochs = 50
    # load the model
    steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model = UNet().to(device)
    print(model)

    train_loader, test_loader = create_mnist_dataloaders(batch_size=128, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)
    
    # Posterior variance for reverse process
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    if os.path.exists(f"model_{steps}.pth"):
        model.load_state_dict(torch.load(f"model_{steps}.pth"))
    else:
        print("No model found, training from scratch")
        model.train()
        time_start = time.time()
        for epoch in range(train_epochs):
            total_loss = 0
            for _, (data, _) in enumerate(train_loader):
                x_0 = data.to(device)
                optimizer.zero_grad()
                t = torch.randint(0, steps, (data.size(0), )).to(device)

                sqrt_alpha_cumprod = sqrt_alphas_cumprod[t][:, None, None, None]
                sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

                noise = torch.randn_like(x_0)
                x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
                    
                predicted = model(x_t, t)
                # print(predicted.max(), predicted.min())
                loss = criterion(predicted, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}")
        # torch.save(model.state_dict(), f"model_{steps}.pth")
        time_end = time.time()
        print(f"Training time: {time_end - time_start} seconds")

    model.eval()

    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            # Start with pure noise
            x_t = torch.randn_like(data).to(device)
            
            for i in range(steps-1, -1, -1):
                # print(i)
                t = torch.full((data.size(0),), i, device=device, dtype=torch.long)
                betas_t = betas[t][:, None, None, None]
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
                sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t])[:, None, None, None]
                
                # 预测噪声
                predicted_noise = model(x_t, t)

                # print(predicted_noise.max(), predicted_noise.min())
                # 计算均值
                model_mean = sqrt_recip_alphas_t * (
                    x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
                )

                if i == 0:
                    x_t = model_mean
                else:
                    posterior_variance_t = posterior_variance[t][:, None, None, None]
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + torch.sqrt(posterior_variance_t) * noise

            from torchvision.utils import save_image
            save_image(x_t[:100], 'ddpm_samples.png', nrow=10, normalize=True)
            print("samples saved to ddpm_samples.png")
            break


if __name__ == "__main__":
    main()
