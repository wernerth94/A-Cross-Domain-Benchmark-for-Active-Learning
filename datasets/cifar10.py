from typing import Tuple, Union, Callable
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from core.data import BaseDataset, normalize, postprocess_torch_dataset
from core.classifier import DenseModel
import requests

class Cifar10(BaseDataset):

    def __init__(self, budget=2000, initial_points_per_class=1, classifier_batch_size=43,
                 data_file="cifar10_al.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, cache_folder)


    def _download_data(self):
        train = torchvision.datasets.CIFAR10(root=self.cache_folder, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=self.cache_folder, train=False, download=True)
        self.x_train, self.y_train, self.x_test, self.y_test = postprocess_torch_dataset(train, test)
        print("Download successful")


    def _normalize_data(self):
        # normalize pixel values from [0..255] to [-1..1]
        high = 255.0
        self.x_train = self.x_train / (high / 2.0) - 1.0
        self.x_test = self.x_test / (high / 2.0) - 1.0



    def get_classifier(self, hidden_dims :Tuple[int] =(24, 12)) -> nn.Module:
        raise NotImplementedError()
        input_size = self.x_test.size(1)
        model = DenseModel(input_size=input_size,
                           num_classes=self.y_test.size(1),
                           hidden_sizes=hidden_dims)
        return model



    def get_optimizer(self, model, lr=0.001, weight_decay=0.0) -> torch.optim.Optimizer:
        raise NotImplementedError()
        return torch.optim.NAdam(model.parameters(), lr=0.001,
                                 weight_decay=0.0)

