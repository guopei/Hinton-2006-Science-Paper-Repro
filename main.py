import os
import time
import math

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import PILImage
from PIL import Image 

from model import MLP
from data import create_mnist_dataloaders, visualize_mnist_data

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def main():
    print("Hello from autoencoder!")
    device="cuda"

    train_epochs = 50
    model = MLP(layers=[784, 1000, 1000, 1000, 1000, 784])
    model.to(device)
    print(model)

    train_loader, test_loader = create_mnist_dataloaders(batch_size=512, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

    # load the model
    steps = 100
    betas = torch.linspace(0.0001, 0.02, steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # Posterior variance for reverse process
    posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

    if os.path.exists(f"model_{steps}.pth"):
        model.load_state_dict(torch.load(f"model_{steps}.pth"))
    else:
        print("No model found, training from scratch")
        model.train()
        time_start = time.time()
        for epoch in range(train_epochs):
            total_loss = 0
            for _, (data, _) in enumerate(train_loader):
                x_0 = data.view(data.size(0), -1).to(device)
                optimizer.zero_grad()
                t = torch.randint(0, steps, (data.size(0), 1)).to(device)

                sqrt_alpha_cumprod = sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alphas_cumprod[t]

                noise = torch.randn_like(x_0)
                x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
                    
                predicted = model(x_t, t)
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
            x_t = torch.randn_like(data.view(data.size(0), -1).to(device))
            
            for step in range(steps-1, -1, -1):
                t = torch.full((x_t.size(0), 1), step, device=device, dtype=torch.long)
                
                # Predict noise at time step t
                noise_pred = model(x_t, t)
                
                # Compute the mean of the reverse diffusion process
                # mean = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * noise_pred)
                mean = sqrt_recip_alphas[step] * (x_t - betas[step] / sqrt_one_minus_alphas_cumprod[step] * noise_pred)

                if step > 0:
                    # Add noise during sampling (except for the last step)
                    noise = torch.randn_like(x_t)
                    # Use the proper posterior variance
                    variance = posterior_variance[step]
                    x_t = mean + torch.sqrt(variance) * noise
                else:
                    # Last step: deterministic (no noise)
                    x_t = mean

            images = visualize_mnist_data(x_t[:100])
            Image.fromarray(images).save(f"outputs_{steps}.png")
            break


if __name__ == "__main__":
    main()
