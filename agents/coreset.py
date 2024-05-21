from typing import Union, Callable
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader


class Coreset_Greedy(BaseAgent):
    """
    Author: Vikas Desai
    Taken from https://github.com/svdesai/coreset-al
    """

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            candidates = self._embed(x_unlabeled, classifier)
            centers = self._embed(x_labeled, classifier)
            dist = pairwise_distances(candidates.detach().cpu(), centers.detach().cpu(), metric='euclidean')
            dist = np.min(dist, axis=1)
            dist = torch.from_numpy(dist)
            chosen = torch.topk(dist, self.query_size).indices.tolist()
        return sample_ids[chosen]

class Coreset_Raw(BaseAgent):
    """
    Author: Vikas Desai
    Taken from https://github.com/svdesai/coreset-al
    """

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            # candidates = self._embed(x_unlabeled, classifier)
            candidates = x_unlabeled.clone()
            # centers = self._embed(x_labeled, classifier)
            centers = x_labeled.clone()
            dist = pairwise_distances(candidates.detach().cpu(), centers.detach().cpu(), metric='euclidean')
            dist = np.min(dist, axis=1)
            dist = torch.from_numpy(dist)
            chosen = torch.topk(dist, self.query_size).indices.tolist()
        return sample_ids[chosen]

