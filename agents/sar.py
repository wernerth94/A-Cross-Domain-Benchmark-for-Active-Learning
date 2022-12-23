from typing import Union, Callable
import os
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent


class SAR(BaseAgent):

    def __init__(self, file="sar_13_12_22.pth", device=None):
        super().__init__()
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        chkpt_path = os.path.join("agents/checkpoints", file)
        if not os.path.exists(chkpt_path):
            raise ValueError(f"Checkpoint {chkpt_path} does not exist")
        agent = torch.load(chkpt_path, map_location=device)
        agent.model.device = device
        agent.model = agent.model.to(device)
        agent.device = device
        self.agent = agent.to(device)

    @classmethod
    def create_state_callback(cls, state_ids: list[int],
                              x_unlabeled: Tensor,
                              x_labeled: Tensor, y_labeled: Tensor,
                              per_class_instances: dict,
                              budget:int, added_images:int,
                              initial_test_acc:float, current_test_acc:float,
                              classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:
        with torch.no_grad():
            sample_x = x_unlabeled[state_ids]
            sample_features = SAR._get_sample_features(sample_x, classifier, y_labeled.shape[1])
            interal_features = SAR._get_internal_features(initial_test_acc, current_test_acc, added_images, budget)
            interal_features = interal_features.unsqueeze(0).repeat(len(sample_features), 1)
            state = torch.cat([sample_features, interal_features], dim=1)
            state = state.cpu()
        return state

    @classmethod
    def _get_internal_features(cls, initial_test_acc, current_test_accuracy, added_images, budget):
        current_acc = torch.Tensor([current_test_accuracy]).cpu()
        improvement = torch.Tensor([current_test_accuracy - initial_test_acc]).cpu()
        avrg_improvement = torch.divide(improvement, max(1, added_images))
        progress = torch.Tensor([added_images / float(budget)]).cpu()
        return torch.cat([current_acc, improvement, avrg_improvement, progress])

    @classmethod
    def _get_sample_features(cls, x, classifier, n_classes):
        eps = 1e-7
        # prediction metrics
        pred = classifier(x).detach()
        pred = torch.softmax(pred, dim=1)
        two_highest, _ = pred.topk(2, dim=1)

        entropy = -torch.mean(pred * torch.log(eps + pred) + (1+eps-pred) * torch.log(1+eps-pred), dim=1)
        bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
        hist_list = [torch.histc(p, bins=10, min=0, max=1) for p in pred]
        hist = torch.stack(hist_list, dim=0) / n_classes

        state = torch.cat([
            bVsSB.unsqueeze(1),
            entropy.unsqueeze(1),
            hist
        ], dim=1)
        return state.cpu()




    def predict(self, state:Union[Tensor, dict], greed:float=0.0) ->Tensor:
        candidates = state["unlabeled_features"]
        centers = state["labeled_features"]
        dist = pairwise_distances(candidates, centers, metric='euclidean')
        dist = np.min(dist, axis=1).reshape(-1, 1)
        dist = torch.from_numpy(dist)
        return torch.argmax(dist, dim=0)