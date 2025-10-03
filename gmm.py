import os
import time

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import PILImage
from PIL import Image 
from sklearn.mixture import GaussianMixture
import numpy as np

from model import MLP
from data import create_mnist_dataloaders, visualize_mnist_data

def main():
    print("Hello from offline GMM!")
    device="cuda"

    model = MLP(encoder_layers=[784, 1000, 500, 250, 30], decoder_layers=[30, 250, 500, 1000, 784]).to(device)
    train_loader, test_loader = create_mnist_dataloaders(batch_size=512, num_workers=4)

    criterion = nn.MSELoss()

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))

    model.eval()
    total_loss = 0
    all_data = []
    all_emb = []
    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            data = data.view(data.size(0), -1).to(device)
            outputs, emb = model(data)
            loss = criterion(outputs, data)
            total_loss += loss.item()

            all_data.append(data.detach().cpu())
            all_emb.append(emb.detach().cpu())

        print(f"Test Loss: {total_loss/len(test_loader)}")
        print(f"Test average square error: {total_loss/len(test_loader)*28*28}")

    all_data = torch.cat(all_data, dim=0)
    all_emb = torch.cat(all_emb, dim=0)

    gmm = GaussianMixture(n_components=10, covariance_type='full')
    gmm.fit(all_emb)

    print(gmm.means_.shape)
    print(gmm.covariances_.shape)
    print(gmm.weights_.shape)

    # Generate n_new samples
    n_new = 100
    new_samples = gmm.sample(n_new)[0]  # [0] gets just the samples, not log-probabilities  

    new_samples = torch.tensor(new_samples, dtype=torch.float32)
    decoded_data = model.decode(new_samples.to(device))
    images = visualize_mnist_data(decoded_data.detach().cpu())
    Image.fromarray(images).save("gmm_decoder_data.png")

if __name__ == "__main__":
    main()
