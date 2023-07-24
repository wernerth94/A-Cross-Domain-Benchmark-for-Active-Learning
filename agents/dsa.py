from typing import Union
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from core.agent import BaseAgent


class DSA(BaseAgent):

    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer,
                      sample_size=100) ->Union[int, list[int]]:
        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        #fetch_dsa(model, x_train, x_target, target_name, layer_names, args)
        # train_ats, train_pred, target_ats, target_pred = self._get_train_target_ats(
        #     classifier, x_train, x_target, target_name
        # )
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


    def _embed(self, x: Tensor, model: Module) -> Tensor:
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
