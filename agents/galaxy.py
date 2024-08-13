#########################################################
# All Credits to Jifan Zhang
# Code adapted from: https://github.com/jifanz/GALAXY
#########################################################
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from scipy.spatial.distance import cdist

def hac(clusters, c_dists):
    num_empty = 0
    while c_dists.shape[0] / float(len(clusters) - num_empty) < 2:
        num_empty += 1
        num_elem = np.array([float(len(c)) for c in clusters])
        i, j = np.unravel_index(np.argmin(c_dists), c_dists.shape)
        assert num_elem[i] != 0. and num_elem[j] != 0.
        c_dists[i] = (c_dists[i] * num_elem[i] + c_dists[j] * num_elem[j]) / (num_elem[i] + num_elem[j])
        c_dists[:, i] = (c_dists[:, i] * num_elem[i] + c_dists[:, j] * num_elem[j]) / (num_elem[i] + num_elem[j])
        c_dists[j] = float("inf")
        c_dists[:, j] = float("inf")
        clusters[i] = clusters[i] + clusters[j]
        clusters[j] = []
    new_clusters = []
    for c in clusters:
        if len(c) != 0:
            new_clusters.append(c)
    return new_clusters


class Galaxy(BaseAgent):

    def _cluster(self, classifier, x_unlabeled):
        features = self._embed(x_unlabeled, classifier)
        features = features.cpu().float()
        dist = torch.cdist(features, features).numpy()
        features = features.numpy()
        np.random.seed(12345)
        slice = np.random.permutation(dist.shape[0])[:1000]
        c_dists = np.array(dist)[slice, :][:, slice]
        for i in range(c_dists.shape[0]):
            c_dists[i, i] = float("inf")
        clusters = hac([[i] for i in range(c_dists.shape[0])], c_dists)
        centers = []
        for i, c in enumerate(clusters):
            center = np.mean(features[slice[c]], axis=0)
            centers.append(center)
        centers = np.array(centers)
        dist = cdist(features, centers)
        cluster_idxs = np.argmin(dist, axis=1)
        return cluster_idxs


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

        with torch.no_grad():
            cluster_batch_size = int(1.25 * self.query_size)
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled), sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            cluster_idxs = self._cluster(classifier, x_unlabeled)
            pred = self._predict(x_unlabeled, classifier)

            sorted_pred = torch.sort(pred, dim=-1, descending=True)[0]
            margins = sorted_pred[:, 0] - sorted_pred[:, 1]
            uncertain_idxs = np.argsort(margins.cpu().numpy())
            clusters = [[] for _ in range(np.max(cluster_idxs) + 1)]
            num_batch_points = 0
            for idx in uncertain_idxs:
                clusters[cluster_idxs[idx]].append(idx)
                num_batch_points += 1
                if num_batch_points == cluster_batch_size:
                    break
            cluster_sizes = np.array([len(c) for c in clusters])
            cluster_sorted_idxs = np.argsort(cluster_sizes)
            c_idx = 0
            chosen = []
            while len(chosen) != self.query_size:
                cluster_idx = cluster_sorted_idxs[c_idx]
                if len(clusters[cluster_idx]) != 0:
                    idx = clusters[cluster_idx].pop(0)
                    chosen.append(idx)
                c_idx = (c_idx + 1) % len(clusters)

        return sample_ids[chosen]

