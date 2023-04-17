from typing import Union, Callable
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent


class Coreset_Greedy(BaseAgent):
    """
    Author: Vikas Desai
    Taken from https://github.com/svdesai/coreset-al
    """

    def predict(self, state_ids: list[int],
                      x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:

        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        with torch.no_grad():
            candidates = classifier._encode(x_unlabeled[state_ids])
            centers = classifier._encode(x_labeled)
            dist = pairwise_distances(candidates.detach().cpu(), centers.detach().cpu(), metric='euclidean')
            dist = np.min(dist, axis=1).reshape(-1, 1)
            dist = torch.from_numpy(dist)
        return torch.argmax(dist, dim=0)
