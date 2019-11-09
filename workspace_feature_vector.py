import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkspaceFeature(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WorkspaceFeature, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.output_layer(x))
        return x