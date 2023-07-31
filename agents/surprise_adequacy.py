from typing import Union
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from core.agent import BaseAgent
from scipy.stats import gaussian_kde
import numpy as np

class LSA(BaseAgent):

    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer,
                      sample_size=100) ->Union[int, list[int]]:
        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        sample_size = min(sample_size, len(x_unlabeled))
        state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=False)

        labeled_embed = self._embed(x_labeled, classifier)
        labeled_pred = classifier(x_labeled)
        unlabeled_pred = classifier(x_unlabeled[state_ids])
        unlabeled_embed = self._embed(x_unlabeled[state_ids], classifier)

        class_matrix = {}
        for label_id in range(len(per_class_instances)):
            class_matrix[label_id] = []
        all_idx = []
        for i, label in enumerate(labeled_pred):
            label = torch.argmax(label).item()
            class_matrix[label].append(i)
            all_idx.append(i)


        kdes, removed_cols = self._get_kdes(labeled_embed, class_matrix)

        max_lsa = -torch.inf
        max_idx = -1 # if not updated, is the same as random sampling
        remaining_cols = list(set(range(unlabeled_embed.size(-1))) - set(removed_cols))
        unlabeled_embed = unlabeled_embed[:, remaining_cols].cpu()
        for i, at in enumerate(unlabeled_embed):
            label = torch.argmax(unlabeled_pred[i]).item()
            kde = kdes[label]
            if kde is None:
                continue
            lsa = -kde.logpdf(at.reshape(-1, 1))
            if lsa > max_lsa:
                max_lsa = lsa
                max_idx = i
        del kdes
        return [state_ids[max_idx]]


    def _get_kdes(self, train_ats, class_matrix, var_threshold=1e-5):
        """Kernel density estimation
        Returns:
            kdes (list): List of kdes per label if classification task.
            removed_cols (list): List of removed columns by variance threshold.
        """
        num_classes = len(class_matrix.keys())
        removed_cols = []
        for label in range(num_classes):
            col_vectors = train_ats[class_matrix[label]].T
            for i in range(col_vectors.shape[0]):
                if (
                    torch.var(col_vectors[i]) < var_threshold
                    and i not in removed_cols
                ):
                    removed_cols.append(i)

        remaining_cols = list(set(range(train_ats.size(-1))) - set(removed_cols))
        train_ats = train_ats[:, remaining_cols].cpu()
        kdes = {}
        for label in range(num_classes):
            refined_ats = train_ats[class_matrix[label]].T
            try:
                kdes[label] = gaussian_kde(refined_ats)
            except np.linalg.LinAlgError: # lower dim. subspace error
                kdes[label] = None
            except ValueError: # less data than features error
                kdes[label] = None
        return kdes, removed_cols

class DSA(BaseAgent):

    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer,
                      sample_size=100) ->Union[int, list[int]]:
        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        sample_size = min(sample_size, len(x_unlabeled))
        state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=False)

        labeled_pred = classifier(x_labeled)
        labeled_embed = self._embed(x_labeled, classifier)
        unlabeled_pred = classifier(x_unlabeled[state_ids])
        unlabeled_embed = self._embed(x_unlabeled[state_ids], classifier)

        class_matrix = {}
        for label_id in range(len(per_class_instances)):
            class_matrix[label_id] = []
        all_idx = []
        for i, label in enumerate(labeled_pred):
            label = torch.argmax(label).item()
            class_matrix[label].append(i)
            all_idx.append(i)

        min_dsa = -torch.inf
        min_idx = None
        for i, at in enumerate(unlabeled_embed):
            label = torch.argmax(unlabeled_pred[i]).item()
            points_of_same_class = class_matrix[label]
            if len(points_of_same_class) == 0:
                dsa = 0.0
            else:
                a_dist, a_dot = self._find_closest_at(at, labeled_embed[class_matrix[label]])
                rest_of_points = list(set(all_idx) - set(class_matrix[label]))
                if len(rest_of_points) > 0:
                    b_dist, _ = self._find_closest_at(a_dot, labeled_embed[rest_of_points])
                    if b_dist.item() == 0:
                        print("b_dist of 0 discovered (Duplicate point or collapsed embeddings)")
                        print("a_dot"); print(a_dot)
                        print("b_dot"); print(_)
                    dsa = a_dist / b_dist
                else:
                    dsa = 0.0
            if dsa > min_dsa:
                min_dsa = dsa
                min_idx = i

        return [state_ids[min_idx],]


    def _find_closest_at(self, at:Tensor, train_ats:Tensor):
        """The closest distance between subject AT and training ATs.

        Args:
            at (list): List of activation traces of an input.
            train_ats (list): List of activation traces in training set (filtered)

        Returns:
            dist (int): The closest distance.
            at (list): Training activation trace that has the closest distance.
        """

        dist = torch.linalg.norm(at - train_ats, axis=1)
        return (torch.min(dist), train_ats[torch.argmin(dist)])
