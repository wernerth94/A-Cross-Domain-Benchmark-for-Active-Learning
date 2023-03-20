from typing import Union, Callable
from abc import ABC, abstractmethod

import torch.nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

class BaseAgent(ABC):

    def __init__(self, agent_rng):
        self.agent_rng = agent_rng
        self.name = str(self.__class__).split('.')[-1][:-2]
        print(f"Loaded Agent: {self.name}")


    @abstractmethod
    def predict(self, state_ids:list[int],
                      x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer)->Union[Tensor, dict]:
        pass


    def get_meta_data(self)->str:
        return f"{self.name}"
