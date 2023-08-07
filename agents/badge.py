import pdb
from copy import deepcopy
from typing import Union
from scipy import stats
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader


# Adapted from
# https://github.com/JordanAsh/badge
# and https://github.com/cure-lab/deep-active-learning
class Badge(BaseAgent):
    def __init__(self, agent_seed, config):
        super().__init__(agent_seed, config)
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(agent_seed)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=100) -> Union[int, list[int]]:

        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        sample_size = min(len(x_unlabeled), sample_size)
        state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=False)

        gradEmbedding = self._get_grad_embedding(x_unlabeled[state_ids], classifier, sample_size)
        chosen = self._init_centers(gradEmbedding, 1)[0]
        return state_ids[chosen].item()


    def _get_grad_embedding(self, X, model, sample_size):
        """ gradient embedding (assumes cross-entropy loss) of the last layer"""
        if isinstance(model, nn.DataParallel):
            model = model.module
        model_mode = model.training
        model.eval()
        all_embeddings = None
        batch_size = min(sample_size, 128)
        loader_te = DataLoader(TensorDataset(X),
                               batch_size=batch_size,
                               generator=self.data_loader_rng)
        device = X.device
        with torch.no_grad():
            for x in loader_te:
                x = x[0]
                x_embed = self._embed(x, model)
                logits = self._predict(x, model)
                batch_probs = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(batch_probs,1)
                emb_dim = x_embed.size(-1)
                n_classes = logits.size(-1)

                if all_embeddings is None:
                    all_embeddings = torch.zeros([0, emb_dim * n_classes])
                batch_embedding = torch.zeros([len(x), emb_dim * n_classes])
                for j in range(len(x)):
                    for c in range(n_classes):
                        if c == predicted_class[j]:
                            batch_embedding[j][emb_dim * c : emb_dim * (c+1)] = deepcopy(x_embed[j]) * (1 - batch_probs[j][c])
                        else:
                            batch_embedding[j][emb_dim * c : emb_dim * (c+1)] = deepcopy(x_embed[j]) * (-1 * batch_probs[j][c])
                all_embeddings = torch.cat([all_embeddings, batch_embedding], dim=0)

            model.training = model_mode
            return all_embeddings


    def _init_centers(self, X, K):
        ind = torch.argmax(torch.linalg.norm(X, dim=1)).item()
        X_arr = X.cpu().numpy()
        mu = [X_arr[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X_arr)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X_arr, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X_arr, [mu[-1]]).ravel().astype(float)
                for i in range(len(X_arr)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X_arr[ind])
            indsAll.append(ind)
            cent += 1
        # gram = np.matmul(X[indsAll], X[indsAll].T)
        # val, _ = np.linalg.eig(gram)
        # val = np.abs(val)
        # vgt = val[val > 1e-2]
        return indsAll


class BatchBadge(Badge):
    def __init__(self, agent_seed, config, batch_size=100):
        super().__init__(agent_seed, config)
        self.batch_size = batch_size


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=8000) -> Union[int, list[int]]:

        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        sample_size = min(len(x_unlabeled), sample_size)
        state_ids = self.agent_rng.choice(len(x_unlabeled), sample_size, replace=False)

        gradEmbedding = self._get_grad_embedding(x_unlabeled[state_ids], classifier, sample_size)
        chosen = self._init_centers(gradEmbedding, self.batch_size)
        return list(state_ids[chosen])
