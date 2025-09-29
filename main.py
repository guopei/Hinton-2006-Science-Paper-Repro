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

    train_epochs = 100
    model = MLP(encoder_layers=[784, 1000, 500, 250, 30], decoder_layers=[30, 250, 500, 1000, 784]).to(device)

    train_loader, test_loader = create_mnist_dataloaders(batch_size=512, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, train_epochs // 10))

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    else:
        # load the model
        model.train()
        time_start = time.time()
        for epoch in range(train_epochs):
            total_loss = 0
            for _, (data, _) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs, _ = model(data.view(data.size(0), -1).to(device))
                loss = criterion(outputs, data.view(data.size(0), -1).to(device))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()               
                total_loss += loss.item()
            scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}")
        time_end = time.time()
        print(f"Training time: {time_end - time_start} seconds")
        torch.save(model.state_dict(), "model.pth")
        print("Model saved to model.pth")

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            outputs, _ = model(data.view(data.size(0), -1).to(device))
            data = data.view(data.size(0), -1).to(device)
            loss = criterion(outputs, data.view(data.size(0), -1).to(device))
            total_loss += loss.item()
        print(f"Test Loss: {total_loss/len(test_loader)}")
        print(f"Test average square error: {total_loss/len(test_loader)*28*28}")

        images = visualize_mnist_data(outputs[:100])
        Image.fromarray(images).save("outputs.png")

        images = visualize_mnist_data(data[:100])
        Image.fromarray(images).save("data.png")

if __name__ == "__main__":
    main()
