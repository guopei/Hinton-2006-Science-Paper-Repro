import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        out = self.linear(x)
        out = self.silu(out)
        out = self.batch_norm(out)

        if self.output_size == self.input_size:
            out += x

        return out
        
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        layers[0] += 1
        self.layers = nn.ModuleList([Layer(input_size, output_size) for input_size, output_size in zip(layers[:-1], layers[1:])])
        
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        for layer in self.layers:
            x = layer(x)

        return x
