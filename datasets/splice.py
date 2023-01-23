from typing import Tuple, Union, Callable
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, normalize, postprocess_svm_data
from core.classifier import DenseModel
import requests

class Splice(BaseDataset):

    def __init__(self, budget=900, initial_points_per_class=1, classifier_batch_size=43,
                 data_file="splice_al.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, cache_folder)


    def _download_data(self):
        train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice"
        test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice.t"

        train_file = os.path.join(self.cache_folder, "splice_train.txt")
        test_file = os.path.join(self.cache_folder, "splice_test.txt")
        with open(train_file, 'w') as f:
            r = requests.get(train_url)
            f.writelines(r.content.decode("utf-8"))
        with open(test_file, 'w') as f:
            r = requests.get(test_url)
            f.writelines(r.content.decode("utf-8"))
        del r

        if os.path.exists(train_file) and os.path.exists(test_file):
            train = load_svmlight_file(train_file, n_features=60)
            test = load_svmlight_file(test_file, n_features=60)
            self.x_train, self.y_train, self.x_test, self.y_test = postprocess_svm_data(train, test)
            print("Download successful")


    def _normalize_data(self):
        self.x_train, self.x_test = normalize(self.x_train, self.x_test, mode="min_max")


    def get_classifier(self, hidden_dims :Tuple[int] =(24, 12)) -> nn.Module:
        input_size = self.x_test.size(1)
        model = DenseModel(input_size=input_size,
                           num_classes=self.y_test.size(1),
                           hidden_sizes=hidden_dims)
        return model



    def get_optimizer(self, model, lr=0.001, weight_decay=0.0) -> torch.optim.Optimizer:
        return torch.optim.NAdam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

