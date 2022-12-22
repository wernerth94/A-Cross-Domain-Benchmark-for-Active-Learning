from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from core.classifier import DenseModel

class MarginScore(BaseAgent):

    @classmethod
    def create_state_callback(cls, state_ids:list[int],
                              x_unlabeled:Tensor, y_unlabeled:Tensor,
                              x_labeled:Tensor, y_labeled:Tensor,
                              per_class_instances:dict,
                              classifier:Module, optimizer:Optimizer) -> Union[Tensor, dict]:
        with torch.no_grad():
            x_sample = x_unlabeled[state_ids]
            pred = classifier(x_sample).detach()
            pred = torch.softmax(pred, dim=1)
            two_highest, _ = pred.topk(2, dim=1)
            bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
        return torch.unsqueeze(bVsSB, dim=-1)


    def predict(self, state: Union[Tensor, dict], greed:float=0.0) -> Tensor:
        return torch.argmax(state, dim=0)