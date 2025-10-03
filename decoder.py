import torch
from data import visualize_mnist_data
from PIL import Image
from model import MLP
from data import create_mnist_dataloaders

def main():
    device = "cuda"
    n_new = 100
    new_samples = torch.randn(n_new, 30).to(device)

    model = MLP(encoder_layers=[784, 1000, 500, 250, 30], decoder_layers=[30, 250, 500, 1000, 784])
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()

    test_loader, _ = create_mnist_dataloaders(batch_size=100, num_workers=4)

    # Random samples
    decoded_data = model.decode(new_samples)
    images = visualize_mnist_data(decoded_data.detach().cpu())
    Image.fromarray(images).save("random_decoder_data.png")

    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            data = data.view(data.size(0), -1).to(device)
            outputs, emb = model(data)

            break
        decoded_data = model.decode(emb)
        images = visualize_mnist_data(decoded_data.detach().cpu())
        Image.fromarray(images).save(f"decoder_data.png")

        # add some noise to the emb
        emb = emb + torch.randn_like(emb) * 0.1
        decoded_data = model.decode(emb)
        images = visualize_mnist_data(decoded_data.detach().cpu())
        Image.fromarray(images).save(f"noisy_decoder_data.png")


if __name__ == "__main__":
    main()