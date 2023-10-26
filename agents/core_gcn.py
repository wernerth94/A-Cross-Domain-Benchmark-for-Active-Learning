################################################################################
# This is an implementation of the paper: Sequential GCN for Active Learning
# Implemented by Yu LI, based on the code: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
# Taken from https://github.com/cure-lab/deep-active-learning and adapted for this benchmark
import abc
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch import Tensor
from torch.nn import functional
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer, Adam
from torch.utils.data import TensorDataset, DataLoader

from core.agent import BaseAgent
import matplotlib.pyplot as plt


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        """r
        nfeat: input feature dimension
        nhid: the hidden layer dimension
        nclass: the output dimension
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        # x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat, x), 1)


def aff_to_adj(x: torch.Tensor, y=None):
    device = x.device
    x = x.detach()  # .cpu().numpy()
    adj = torch.matmul(x, x.T)
    adj += -1.0 * torch.eye(adj.shape[0]).to(device)
    adj_diag = torch.sum(adj, dim=0) + 1e-6  # preventing division by 0
    adj = torch.matmul(adj, torch.diag(1 / adj_diag).to(device))
    adj = adj + torch.eye(adj.size(0)).to(device)
    # adj = torch.Tensor(adj)
    return adj


def BCEAdjLoss(scores, labeled_idxs, unlabeled_idxs, l_adj, eps=1e-6):
    # added an epsilon to avoid division log(0)
    lnl = torch.log(scores[labeled_idxs] + eps)
    lnu = torch.log(1 - scores[unlabeled_idxs] + eps)
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def select_batch_unc_(self, **kwargs):
        return self.select_batch_unc_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):

    def __init__(self, X, metric='euclidean'):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        x = self.features[cluster_centers]
        # Update min_distances for all examples given new cluster center.
        dist = pairwise_distances(self.features, x, metric=self.metric)  # ,n_jobs=4)

        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        self.already_selected = already_selected

        return new_batch


class CoreGCN(BaseAgent):
    """
    Author: Cure-Lab
    Taken from https://github.com/cure-lab/deep-active-learning
    """

    def __init__(self, agent_seed, config: dict, query_size=1,
                 gcn_n_hidden=128, gcn_dropout=0.3, gcn_lambda=1.2):
        super().__init__(agent_seed, config, query_size)
        self.gcn_n_hidden = gcn_n_hidden
        self.gcn_dropout = gcn_dropout
        self.gcn_lambda = gcn_lambda

    def predict(self, x_unlabeled: Tensor,
                x_labeled: Tensor, y_labeled: Tensor,
                per_class_instances: dict,
                budget: int, added_images: int,
                initial_test_acc: float, current_test_acc: float,
                classifier: Module, optimizer: Optimizer,
                sample_size=5000) -> Union[int, list[int]]:

        assert hasattr(classifier, "_encode"), "The provided model needs the '_encode' function"
        sample_size = min(sample_size, len(x_unlabeled))
        sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
        x_unlabeled = x_unlabeled[sample_ids]

        device = x_unlabeled.device
        all_points = torch.cat([x_unlabeled, x_labeled], dim=0)
        features = self._embed(all_points, classifier)
        features = functional.normalize(features)
        adj = aff_to_adj(features)

        gcn = GCN(nfeat=features.shape[1],
                  nhid=self.gcn_n_hidden,
                  nclass=1,
                  dropout=self.gcn_dropout).to(device)

        optim = Adam(gcn.parameters(), lr=1e-3,
                     weight_decay=5e-4)

        lbl = np.arange(len(x_unlabeled), len(x_unlabeled) + len(x_labeled), 1)  # temp labeled index
        nlbl = np.arange(0, len(x_unlabeled), 1)  # temp unlabled index

        # train the gcn model
        losses = []  # for debugging
        for _ in range(200):
            optim.zero_grad()
            outputs, _, _ = gcn(features, adj)
            lamda = self.gcn_lambda
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        gcn.eval()
        with torch.no_grad():
            inputs = features
            scores, _, feat = gcn(inputs, adj)

            feat = feat.detach().cpu().numpy()
            new_av_idx = np.arange(len(x_unlabeled), len(x_unlabeled) + len(x_labeled))
            sampling2 = kCenterGreedy(feat)
            selected_indices = sampling2.select_batch_(new_av_idx, 1)

        selected_indices = list(selected_indices)
        return sample_ids[selected_indices]

