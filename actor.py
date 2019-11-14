import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = []
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output_layer = nn.Linear(hidden_sizes[len(hidden_sizes) - 1], output_size)

    def forward(self, state):
        x = F.elu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        x = torch.tanh(self.output_layer(x))

        return x