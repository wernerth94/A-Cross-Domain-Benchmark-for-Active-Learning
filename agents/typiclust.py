from typing import Union
import numpy as np
import pandas as pd
import faiss
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from core.agent import BaseAgent
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer




class TypiClust(BaseAgent):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer)->Union[Tensor, dict]:

        num_clusters = min(len(x_labeled) + 1, self.MAX_NUM_CLUSTERS)
        all_data = torch.concat([x_unlabeled, x_labeled], dim=0).cpu()
        clusters = self._kmeans(all_data, num_clusters=num_clusters)
        labels = np.copy(clusters)
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_ids, cluster_sizes = self._fill_in_zero_size_clusters(cluster_ids, cluster_sizes)
        existing_indices = np.arange(len(x_unlabeled), len(x_unlabeled)+len(x_labeled))
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids,
                                    'cluster_size': cluster_sizes,
                                    'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]

        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        # pick the most typical example from the top-ranked cluster
        cluster_id = clusters_df.iloc[0].cluster_id
        indices = (labels == cluster_id).nonzero()[0]
        rel_feats = all_data[indices].numpy()
        typicality = self._calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
        idx = indices[typicality.argmax()]
        return idx


    def _get_nn(self, features:np.ndarray, num_neighbors:int):
        # calculates nearest neighbors on GPU
        d = features.shape[1]
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(features.astype(np.float32))  # add vectors to the index
        distances, indices = cpu_index.search(features, num_neighbors + 1)
        # 0 index is the same sample, dropping it
        return distances[:, 1:], indices[:, 1:]


    def _get_mean_nn_dist(self, features, num_neighbors, return_indices=False):
        distances, indices = self._get_nn(features, num_neighbors)
        mean_distance = distances.mean(axis=1)
        if return_indices:
            return mean_distance, indices
        return mean_distance


    def _calculate_typicality(self, features, num_neighbors):
        mean_distance = self._get_mean_nn_dist(features, num_neighbors)
        # low distance to NN is high density
        typicality = 1 / (mean_distance + 1e-5)
        return typicality


    def _kmeans(self, features, num_clusters):
        random_state = self.agent_rng.integers(10000)
        if num_clusters <= 50:
            km = KMeans(n_clusters=num_clusters, random_state=random_state,
                        n_init=10) # suppressing deprecated warning for n_init
            km.fit_predict(features)
        else:
            km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, random_state=random_state,
                                 n_init=10)  # suppressing deprecated warning for n_init
            km.fit_predict(features)
        return km.labels_


    def _fill_in_zero_size_clusters(self, clusters:np.ndarray, counts:np.ndarray):
        i = 1
        while i < len(clusters):
            if clusters[i] != i:
                clusters = np.insert(clusters, i, i)
                counts = np.insert(counts, i, 0)
            i += 1
        return clusters, counts
