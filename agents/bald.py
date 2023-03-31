from typing import Union
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent

class BALD(BaseAgent):

    def __init__(self, agent_rng, dropout_trials=5):
        super().__init__(agent_rng)
        self.dropout_trials = dropout_trials


    def predict(self, state_ids: list[int],
                      x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:

        with torch.no_grad():
            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            x_sample = x_unlabeled[state_ids]
            y_hat_sum = torch.zeros( (len(x_sample), y_labeled.size(-1)) )
            entropy_sum = torch.zeros( (len(x_sample)) )
            for trial in range(self.dropout_trials):
                y_hat = classifier(x_sample)
                y_hat = torch.nn.functional.softmax(y_hat, dim=-1)
                y_hat_sum += y_hat

                y_hat_log = torch.log2(y_hat + 1e-6)#Add 1e-6 to avoid log(0)
                entropy_matrix = -torch.multiply(y_hat, y_hat_log)
                entropy_per_instance = torch.sum(entropy_matrix, dim=1)
                entropy_sum += entropy_per_instance

            avg_pi = torch.divide(y_hat_sum, self.dropout_trials)
            log_avg_pi = torch.log2(avg_pi + 1e-6)
            entropy_avg_pi = -torch.multiply(avg_pi, log_avg_pi)
            entropy_avg_pi = torch.sum(entropy_avg_pi, dim=1)
            g_x = entropy_avg_pi
            avg_entropy = torch.divide(entropy_sum, self.dropout_trials)
            f_x = avg_entropy

            u_x = g_x - f_x

        return torch.argmax(u_x, dim=0)
