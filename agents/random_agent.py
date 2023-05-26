from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent

class RandomAgent(BaseAgent):

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:
        idx = int(self.agent_rng.random()*len(x_unlabeled))
        return idx

class BatchRandomAgent(BaseAgent):
    def __init__(self, agent_seed, config:dict, batch_size=10):
        super().__init__(agent_seed, config)
        self.batch_size = batch_size

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:
        ids = self.agent_rng.choice(len(x_unlabeled), self.batch_size, replace=False)
        return ids
