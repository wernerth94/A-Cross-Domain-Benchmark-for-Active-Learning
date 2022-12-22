from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent

class RandomAgent(BaseAgent):

    @classmethod
    def create_state_callback(cls, state_ids:list[int],
                              x_unlabeled:Tensor, y_unlabeled:Tensor,
                              x_labeled:Tensor, y_labeled:Tensor,
                              per_class_instances:dict,
                              classifier:Module, optimizer:Optimizer) -> Union[Tensor, dict]:
        s = np.array([state_ids]).T
        return torch.from_numpy(s)


    def predict(self, state: Union[Tensor, dict], greed:float=0.0) -> Tensor:
        return torch.randint(len(state), size=(1,1))