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
                      sample_size=8000) -> Union[Tensor, dict]:

        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=False)
            candidates = self._embed(x_unlabeled[state_ids], classifier)
            centers = self._embed(x_labeled, classifier)
            dist = pairwise_distances(candidates.detach().cpu(), centers.detach().cpu(), metric='euclidean')
            dist = np.min(dist, axis=1).reshape(-1, 1)
            dist = torch.from_numpy(dist)
        return state_ids[torch.argmax(dist, dim=0)].item()

    def _embed(self, x:Tensor, model:Module)->Tensor:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(x),
                                batch_size=256)
            emb_x = None
            for batch in loader:
                batch = batch[0]
                emb_batch = model._encode(batch)
                if emb_x is None:
                    emb_dim = emb_batch.size(-1)
                    emb_x = torch.zeros((0, emb_dim)).to(emb_batch.device)
                emb_x = torch.cat([emb_x, emb_batch])
        return emb_x
