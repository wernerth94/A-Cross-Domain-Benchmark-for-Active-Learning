from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent

class MarginScore(BaseAgent):

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=100) -> Union[Tensor, dict]:

        with torch.no_grad():
            replacement_needed = len(x_unlabeled) < sample_size
            state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=replacement_needed)
            x_sample = x_unlabeled[state_ids]
            pred = classifier(x_sample).detach()
            pred = torch.softmax(pred, dim=1)
            two_highest, _ = pred.topk(2, dim=1)
            bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
            torch.unsqueeze(bVsSB, dim=-1)
        return state_ids[torch.argmax(bVsSB, dim=0)]
