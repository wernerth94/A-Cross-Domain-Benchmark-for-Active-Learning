from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from batchbald_redux.batchbald import get_batchbald_batch

# Based on https://github.com/acl21/deep-active-learning-pytorch
# Author: Akshay L Chandra
class BALD(BaseAgent):

    def __init__(self, agent_seed, config, query_size=1, dropout_trials=25):
        super().__init__(agent_seed, config, query_size)
        assert "current_run_info" in config and "encoded" in config["current_run_info"]
        self.dropout_trials = dropout_trials

    @classmethod
    def inject_config(cls, config:dict):
        """
        Add dropout to classification model
        """
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = 0.3


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            x_sample = x_unlabeled
            y_hat_sum = torch.zeros( (len(x_sample), y_labeled.size(-1)) ).to(device)
            entropy_sum = torch.zeros(len(x_sample)).to(device)
            y_hat_collection = torch.zeros( (len(x_sample), self.dropout_trials, y_labeled.size(-1)) ).to(device)
            for trial in range(self.dropout_trials):
                y_hat = self._predict(x_sample, classifier)
                y_hat = torch.nn.functional.softmax(y_hat, dim=-1)
                y_hat_collection[:, trial, :] = y_hat
            #     y_hat_sum += y_hat
            #
            #     y_hat_log = torch.log2(y_hat + 1e-6)  # Add 1e-6 to avoid log(0)
            #     entropy_matrix = -torch.multiply(y_hat, y_hat_log)
            #     entropy_per_instance = torch.sum(entropy_matrix, dim=1)
            #     entropy_sum += entropy_per_instance
            #
            # avg_pi = torch.divide(y_hat_sum, self.dropout_trials)
            # log_avg_pi = torch.log2(avg_pi + 1e-6)
            # entropy_avg_pi = -torch.multiply(avg_pi, log_avg_pi)
            # entropy_avg_pi = torch.sum(entropy_avg_pi, dim=1)
            # g_x = entropy_avg_pi
            # avg_entropy = torch.divide(entropy_sum, self.dropout_trials)
            # f_x = avg_entropy
            # u_x = g_x - f_x

            # taken from https://github.com/cure-lab/deep-active-learning/blob/main/query_strategies/batch_BALD.py
        # return torch.topk(u_x, self.query_size).indices.tolist()
            res = get_batchbald_batch(y_hat_collection, self.query_size, 100)
        ids = res.indices
        return sample_ids[ids]

