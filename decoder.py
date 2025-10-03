import torch
from data import visualize_mnist_data
from PIL import Image
from model import MLP

def main():
    device = "cuda"
    n_new = 100
    new_samples = torch.randn(n_new, 30).to(device)

    model = MLP(encoder_layers=[784, 1000, 500, 250, 30], decoder_layers=[30, 250, 500, 1000, 784])
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()

    decoded_data = model.decode(new_samples)
    images = visualize_mnist_data(decoded_data.detach().cpu())
    Image.fromarray(images).save("random_decoder_data.png")


if __name__ == "__main__":
    main()