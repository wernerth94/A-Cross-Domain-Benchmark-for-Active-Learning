from typing import Union, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader

class ShannonEntropy(BaseAgent):

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=8000) -> Union[int, list[int]]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=False)
            x_sample = x_unlabeled[state_ids]
            pred = self._predict(x_sample, classifier)
            pred = torch.softmax(pred, dim=1)
            eps = 1e-7
            entropy = -torch.mean(pred * torch.log(eps + pred) + (1 + eps - pred) * torch.log(1 + eps - pred), dim=1)
            entropy = torch.unsqueeze(entropy, dim=-1)
        return state_ids[torch.argmax(entropy, dim=0)].item()

    def _predict(self, x:Tensor, model:Module)->Tensor:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(x),
                                batch_size=256)
            y_hat = None
            for batch in loader:
                batch = batch[0]
                emb_batch = model(batch)
                if y_hat is None:
                    emb_dim = emb_batch.size(-1)
                    y_hat = torch.zeros((0, emb_dim)).to(emb_batch.device)
                y_hat = torch.cat([y_hat, emb_batch])
        return y_hat
