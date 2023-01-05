import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

# def get_dense_sequential_model(input_size:int, num_classes:int, hidden_sizes:tuple)->nn.Sequential:
#     if len(hidden_sizes) > 0:
#         inpt = nn.Linear(input_size, hidden_sizes[0])
#         hidden = list()
#         for i in range(len(hidden_sizes)):
#             hidden.append(nn.Linear(hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
#             hidden.append(nn.ReLU(inplace=True))
#         out = nn.Linear(hidden_sizes[-1], num_classes)
#         model = nn.Sequential(inpt, nn.ReLU(inplace=True),
#                               *hidden,
#                               out)
#     else:
#         model = nn.Sequential(nn.Linear(input_size, num_classes),
#                               nn.ReLU(inplace=True))
#     example_forward_input = torch.rand(1, input_size)
#     model = torch.jit.trace_module(model, {'forward': example_forward_input})
#     return model

class DenseModel(nn.Module):
    def __init__(self, input_size:int, num_classes:int, hidden_sizes:tuple):
        assert len(hidden_sizes) > 0
        super().__init__()

        self.inpt = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = []
        for i in range(len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
        self.out = nn.Linear(hidden_sizes[-1], num_classes)

    def _encode(self, x:Tensor)->Tensor:
        x = self.inpt(x)
        x = F.relu(x)
        for h_layer in self.hidden:
            x = h_layer(x)
            x = F.relu(x)
        return x

    def forward(self, x):
        x = self._encode(x)
        x = self.out(x)
        return x