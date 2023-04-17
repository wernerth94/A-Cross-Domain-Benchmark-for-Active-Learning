import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    """
    Based on code from TypiClust: https://github.com/avihu111/typiclust
    Hacohen, Guy, Avihu Dekel, and Daphna Weinshall.
    "Active learning on a budget: Opposite strategies suit high and low budgets." arXiv preprint arXiv:2202.02794 (2022).
    Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
    """

    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))


    def forward(self, x, return_pre_last=False):
        pre_last = self.backbone(x)
        features = self.contrastive_head(pre_last)
        features = F.normalize(features, dim = 1)
        if return_pre_last:
            return features, pre_last
        return features



