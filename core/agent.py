from typing import Union
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import numpy as np

class BaseAgent(ABC):

    def __init__(self, agent_seed, config:dict):
        self.agent_seed = agent_seed
        self.agent_rng = np.random.default_rng(agent_seed)
        self.config = config
        self.name = str(self.__class__).split('.')[-1][:-2]
        print(f"Loaded Agent: {self.name}")

    @classmethod
    def inject_config(cls, config:dict):
        """
        This method can be used to change the dataset config.
        I.e. add dropout to the classification model
        """
        pass


    @abstractmethod
    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer)->Union[Tensor, dict]:
        pass


    def get_meta_data(self)->str:
        return f"{self.name}"
