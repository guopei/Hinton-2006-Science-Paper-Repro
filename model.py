import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.elu = nn.ELU()
        self.batch_norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        out = self.linear(x)
        out = self.elu(out)
        out = self.batch_norm(out)
        return out
        
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        layers[0] += 1
        self.first_layer = Layer(layers[0], layers[1])
        self.middle_layers = nn.ModuleList([Layer(layers[i], layers[i+1]) for i in range(1, len(layers) - 2)])
        self.last_layer = Layer(layers[-2], layers[-1])

    def forward(self, x, t):
        x = torch.cat((x, t), -1)
        x = self.first_layer(x)
        for layer in self.middle_layers:
            residual = layer(x)
            x = x + residual
        x = self.last_layer(x)
        return x

    def step(self, x, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x.shape[0], 1)
        # Use simple Euler method for stability
        return x + (t_end - t_start) * self(x, t_start)