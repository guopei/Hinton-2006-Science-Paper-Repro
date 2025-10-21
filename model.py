import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        out = self.batch_norm(out)
        return out
        
class MLP(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super(MLP, self).__init__()
        self.encoder_layers = nn.ModuleList([Layer(input_size, output_size) for input_size, output_size in zip(encoder_layers[:-1], encoder_layers[1:])])
        self.decoder_layers = nn.ModuleList([Layer(input_size, output_size) for input_size, output_size in zip(decoder_layers[:-2], decoder_layers[1:-1])])
        self.mean_layer = nn.Linear(encoder_layers[-1], decoder_layers[0])
        self.std_layer = nn.Linear(encoder_layers[-1], decoder_layers[0])
        self.final_layer = nn.Linear(decoder_layers[-2], decoder_layers[-1])
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.encode(x)

        mean = self.mean_layer(x)
        std = self.std_layer(x)

        z = mean + torch.randn_like(mean) * std

        x = self.decode(z)
        return x, (mean, std)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, emb):
        for layer in self.decoder_layers:
            emb = layer(emb)
        emb = self.final_layer(emb)
        emb = self.sigmoid(emb)
        return emb

    def sample(self, x):

        x = self.decode(x)
        return x


if __name__ == "__main__":
    model = MLP(encoder_layers=[784, 1000, 500, 250, 30], decoder_layers=[30, 250, 500, 1000, 784])
    print(model)