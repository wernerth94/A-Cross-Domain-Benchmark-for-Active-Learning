from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent

class ShannonEntropy(BaseAgent):

    def predict(self, state_ids: list[int],
                      x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:

        with torch.no_grad():
            x_sample = x_unlabeled[state_ids]
            pred = classifier(x_sample).detach()
            pred = torch.softmax(pred, dim=1)
            eps = 1e-7
            entropy = -torch.mean(pred * torch.log(eps + pred) + (1 + eps - pred) * torch.log(1 + eps - pred), dim=1)
            entropy = torch.unsqueeze(entropy, dim=-1)
        return torch.argmax(entropy, dim=0)
