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
        self.final_layer = nn.Linear(decoder_layers[-2], decoder_layers[-1])
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)

        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.sigmoid(x)
        return x
