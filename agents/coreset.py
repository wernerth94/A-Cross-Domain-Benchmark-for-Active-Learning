from typing import Union, Callable
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from core.classifier import DenseCoresetModel


class Coreset_Greedy(BaseAgent):
    """
    Author: Vikas Desai
    Taken from https://github.com/svdesai/coreset-al
    """

    @classmethod
    def get_classifier_factory(cls) -> Callable:
        return DenseCoresetModel

    @classmethod
    def create_state_callback(cls, state_ids: list[int],
                              x_unlabeled: Tensor, y_unlabeled: Tensor,
                              x_labeled: Tensor, y_labeled: Tensor,
                              per_class_instances: dict,
                              classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:
        assert hasattr(classifier, "get_features"), "The provided model needs the 'get_features' function"
        with torch.no_grad():
            labeled_features = classifier.get_features(x_labeled)
            unlabeled_features = classifier.get_features(x_unlabeled[state_ids])
        return {
            "labeled_features" : labeled_features,
            "unlabeled_features": unlabeled_features
        }

    def predict(self, state:Union[Tensor, dict], greed:float=0.0) ->Tensor:
        candidates = state["unlabeled_features"]
        centers = state["labeled_features"]
        dist = pairwise_distances(candidates, centers, metric='euclidean')
        dist = np.min(dist, axis=1).reshape(-1, 1)
        dist = torch.from_numpy(dist)
        return torch.argmax(dist, dim=0)