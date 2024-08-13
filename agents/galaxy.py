#########################################################
# All Credits to Jifan Zhang
# Code adapted from: https://github.com/jifanz/GALAXY
#########################################################

from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader

class Galaxy(BaseAgent):

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled), sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            pred = self._predict(x_unlabeled, classifier)

            sorted_pred = torch.sort(pred, dim=-1, descending=True)[0]
            margins = sorted_pred[:, 0] - sorted_pred[:, 1]
            uncertain_idxs = np.argsort(margins.cpu().numpy())
            clusters = [[] for _ in range(np.max(cluster_idxs) + 1)]
            num_batch_points = 0
            for idx in uncertain_idxs:
                if idx not in queried_set:
                    clusters[cluster_idxs[idx]].append(idx)
                    num_batch_points += 1
                if num_batch_points == cluster_batch_size:
                    break
            cluster_sizes = np.array([len(c) for c in clusters])
            cluster_sorted_idxs = np.argsort(cluster_sizes)
            c_idx = 0
            while len(queried) != self.query_size:
                cluster_idx = cluster_sorted_idxs[c_idx]
                if len(clusters[cluster_idx]) != 0:
                    idx = clusters[cluster_idx].pop(0)
                    queried.append(idx)
                    queried_set.add(idx)
                c_idx = (c_idx + 1) % len(clusters)
            model_acc = []
            batch = torch.from_numpy(np.array(list(queried))).long()

            chosen = torch.topk(bVsSB, self.query_size).indices.tolist()
        return sample_ids[chosen]

