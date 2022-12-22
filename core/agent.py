from typing import Union, Callable
from abc import ABC, abstractmethod

import torch.nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

class BaseAgent(ABC):

    def __init__(self):
        self.name = str(self.__class__).split('.')[-1][:-2]

    @classmethod
    @abstractmethod
    def create_state_callback(cls, state_ids:list[int],
                              x_unlabeled:Tensor, y_unlabeled:Tensor,
                              x_labeled:Tensor, y_labeled:Tensor,
                              per_class_instances:dict,
                              classifier:Module, optimizer:Optimizer)->Union[Tensor, dict]:
        pass

    @abstractmethod
    def predict(self, state:Union[Tensor, dict], greed:float=0.0)->Tensor:
        pass
