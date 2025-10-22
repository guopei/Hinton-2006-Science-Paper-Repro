import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.linear(x)
        out = self.elu(out)
        out = self.dropout(out)
        return out
        
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        layers[0] += 1
        self.layers = nn.ModuleList([Layer(input_size, output_size) for input_size, output_size in zip(layers[:-1], layers[1:])])
        self.elu = nn.ELU()
        
        
    def forward(self, x, t):
        x = torch.cat((x, t), -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x.shape[0], 1)
        # Use simple Euler method for stability
        return x + (t_end - t_start) * self(x, t_start)