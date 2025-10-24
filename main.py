import os
import time

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

    train_epochs = 500
    model = MLP(layers=[784, 6000, 6000, 6000, 6000, 6000, 6000, 784])
    model.to(device)
    print(model)

    train_loader, test_loader = create_mnist_dataloaders(batch_size=512, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate
    criterion = nn.MSELoss()

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, train_epochs // 5))

    # load the model
    if os.path.exists(f"model_{train_epochs}.pth"):
        model.load_state_dict(torch.load(f"model_{train_epochs}.pth"))
    else:
        print("No model found, training from scratch")
        model.train()
        time_start = time.time()
        for epoch in range(train_epochs):
            total_loss = 0
            for _, (data, _) in enumerate(train_loader):
                data = data.view(data.size(0), -1).to(device)
                optimizer.zero_grad()
                noise = torch.randn_like(data)
                
                t = torch.rand(len(data), 1).to(device)

                x_t = (1 - t) * noise + t * data
                dx_t = data - noise

                predicted = model(x_t, t)
                loss = criterion(predicted, dx_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}")
        # torch.save(model.state_dict(), f"model_{train_epochs}.pth")
        time_end = time.time()
        print(f"Training time: {time_end - time_start} seconds")

    model.eval()
    n_steps = 100
    with torch.no_grad():
        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        x = torch.randn(100, 784).to(device)
        for i in range(n_steps):
            x = model.step(x, time_steps[i], time_steps[i + 1])

        print(x.max(), x.min())
        images = visualize_mnist_data(x[:100])
        Image.fromarray(images).save(f"outputs_{n_steps}.png")

if __name__ == "__main__":
    main()
