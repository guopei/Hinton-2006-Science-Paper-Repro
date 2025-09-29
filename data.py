from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):

    preprocess = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    train_dataset = MNIST(
        root="./mnist_data", train=True, download=True, transform=preprocess
    )
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def visualize_mnist_data(images):
    # convert model output to numpy array
    # place them into a grid of n by n

    n = int(np.sqrt(images.size(0)))
    images = images.view(n, n, 28, 28)
    images = images.permute(0, 2, 1, 3)
    images = images.reshape(n*28, n*28)
    images = images.detach().cpu().numpy()
    
    images = images * 255
    images = images.astype(np.uint8)
    
    return images
