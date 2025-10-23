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
from torch.optim.lr_scheduler import LambdaLR

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def linear_warmup(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def main():
    print("Hello from autoencoder!")
    device="cuda"

    train_epochs = 50
    model = MLP(encoder_layers=[784, 1000, 500, 250, 30], decoder_layers=[30, 250, 500, 1000, 784])
    model.to(device)

    train_loader, test_loader = create_mnist_dataloaders(batch_size=512, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, train_epochs // 10))

    # load the model
    steps = 100
    sigma = 0.95
    if os.path.exists(f"model_{steps}.pth"):
        model.load_state_dict(torch.load(f"model_{steps}.pth"))
    else:
        print("No model found, training from scratch")
        model.train()
        time_start = time.time()
        for epoch in range(train_epochs):
            total_loss = 0
            for _, (data, _) in enumerate(train_loader):
                data = data.view(data.size(0), -1).to(device)
                optimizer.zero_grad()
                random_steps = torch.randint(1, steps+1, (data.size(0), 1)).to(device)

                noised_data = torch.sqrt(sigma**random_steps) * data + torch.randn_like(data) * torch.sqrt(1 - sigma**random_steps)
                denoised_data = torch.sqrt(sigma**(random_steps-1)) * data + torch.randn_like(data) * torch.sqrt(1 - sigma**(random_steps-1))
                    
                predicted = model(noised_data)
                loss = criterion(predicted, denoised_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}")
        torch.save(model.state_dict(), f"model_{steps}.pth")
        time_end = time.time()
        print(f"Training time: {time_end - time_start} seconds")

    model.eval()

    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            data = data.view(data.size(0), -1).to(device)
            noise = torch.randn_like(data)
            current_output = noise
            for _ in range(steps):
                current_output = model(current_output)
                current_output += noise * sigma**(_+1)

            images = visualize_mnist_data(current_output[:100])
            Image.fromarray(images).save(f"outputs_{steps}.png")
            break


if __name__ == "__main__":
    main()
