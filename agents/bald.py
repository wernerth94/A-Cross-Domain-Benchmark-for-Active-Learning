from typing import Union
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from core.classifier import construct_model

class BALD(BaseAgent):

    def __init__(self, agent_rng, config, dropout_trials=5):
        super().__init__(agent_rng, config)
        assert "current_run_info" in config and "embedded" in config["current_run_info"]
        self.dropout_trials = dropout_trials
        self.dropout_model = None
        self.dropout_model_rng = torch.Generator()
        self.dropout_model_rng.manual_seed(1)


    def _create_dropout_model(self, x_shape, y_shape):
        if self.config["current_run_info"]["embedded"]:
            self.dropout_model, _ = construct_model(self.dropout_model_rng, x_shape, y_shape[-1], self.config["classifier_embedded"])
        else:
            self.dropout_model, _ = construct_model(self.dropout_model_rng, x_shape, y_shape[-1], self.config["classifier"])
        self.initial_weights = self.dropout_model.state_dict()

    def _fit_dropout_model(self, x, y):
        if self.config["current_run_info"]["embedded"]:
            self.dropout_model.load_state_dict(self.initial_weights)

        drop_last = 64 < len(self.x_labeled)
        train_dataloader = DataLoader(TensorDataset(self.x_labeled, self.y_labeled),
                                      batch_size=self.dataset.classifier_batch_size,
                                      drop_last=drop_last,
                                      generator=self.data_loader_rng,
                                      # num_workers=4, # dropped for CUDA compat
                                      shuffle=True)
        test_dataloader = DataLoader(TensorDataset(self.dataset.x_test, self.dataset.y_test), batch_size=512,
                                     # num_workers=4 # dropped for CUDA compat
                                     )

        lastLoss = torch.inf
        for e in range(epochs):
            for batch_x, batch_y in train_dataloader:
                yHat = self.classifier(batch_x)
                loss_value = self.loss(yHat, batch_y)
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
            # early stopping on test
            with torch.no_grad():
                loss_sum = 0.0
                total = 0.0
                correct = 0.0
                for batch_x, batch_y in test_dataloader:
                    yHat = self.classifier(batch_x)
                    predicted = torch.argmax(yHat, dim=1)
                    total += batch_y.size(0)
                    correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
                    class_loss = self.loss(yHat, torch.argmax(batch_y.long(), dim=1))
                    loss_sum += class_loss.detach().cpu().numpy()
                # early stop on test with patience of 0
                if loss_sum >= lastLoss:
                    break
                lastLoss = loss_sum
        accuracy = correct / total
        self.current_test_loss = loss_sum

        reward = accuracy - self.current_test_accuracy
        self.current_test_accuracy = accuracy


    def predict(self, state_ids: list[int],
                      x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:
        if self.dropout_model is None:
            self._create_dropout_model(x_labeled.size(), y_labeled.size())
        self._fit_dropout_model(x_labeled, y_labeled)

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

                y_hat_log = torch.log2(y_hat + 1e-6)  # Add 1e-6 to avoid log(0)
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
