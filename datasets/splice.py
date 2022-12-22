from typing import Tuple, Union, Callable
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, normalize, postprocess_svm_data
from core.classifier import DenseModel

class Splice(BaseDataset):

    def __init__(self, budget=900, initial_points_per_class=1, classifier_batch_size=43,
                 cache_folder:str="~/.al_benchmark/datasets"):
        super().__init__(budget, initial_points_per_class, classifier_batch_size, cache_folder)


    def _download_data(self):
        raise NotImplementedError("Download is not implemented")

        train_file = os.path.join(self.cache_folder, "splice_train.txt")
        test_file = os.path.join(self.cache_folder, "splice_test.txt")
        if os.path.exists(train_file) and os.path.exists(test_file):
            train = load_svmlight_file(train_file, n_features=60)
            test = load_svmlight_file(test_file, n_features=60)
            x_train, y_train, x_test, y_test = postprocess_svm_data(train, test)
            return x_train, y_train, x_test, y_test


    def _create_seed_set(self):
        nClasses = self.y_train.shape[1]
        x_labeled, y_labeled = [], []

        ids = np.arange(self.x_train.shape[0], dtype=int)
        np.random.shuffle(ids)
        perClassIntances = [0 for _ in range(nClasses)]
        usedIds = []
        for i in ids:
            label = torch.argmax(self.y_train[i])
            if perClassIntances[label] < self.initial_points_per_class:
                x_labeled.append(i)
                y_labeled.append(i)
                usedIds.append(i)
                perClassIntances[label] += 1
            if sum(perClassIntances) >= self.initial_points_per_class * nClasses:
                break
        unusedIds = [i for i in np.arange(self.x_train.shape[0]) if i not in usedIds]
        self.x_labeled = self.x_train[x_labeled]
        self.y_labeled = self.y_train[y_labeled]
        self.x_unlabeled = self.x_train[unusedIds]
        self.y_unlabeled = self.y_train[unusedIds]
        del self.x_train
        del self.y_train

        torch.save({
            "x_labeled": self.x_labeled,
            "y_labeled": self.y_labeled,
            "x_unlabeled": self.x_unlabeled,
            "y_unlabeled": self.y_unlabeled,
            "x_test": self.x_test,
            "y_test": self.y_test,
        }, f"../../datasets/splice_al.pt")
        return True


    def _load_data(self) -> Union[None, Tuple]:
        file = os.path.join(self.cache_folder, "splice_al.pt")
        if os.path.exists(file):
            dataset = torch.load(file)
            return dataset["x_labeled"], dataset["y_labeled"], \
                dataset["x_unlabeled"], dataset["y_unlabeled"], \
                dataset["x_test"], dataset["y_test"]
        return None


    def _normalize_data(self):
        self.x_train, self.x_test = normalize(self.x_train,self.x_test, mode="min_max")


    def get_classifier(self, hidden_dims :Tuple[int] =(24, 12)) -> nn.Module:
        return DenseModel(input_size=self.x_test.size(1),
                          num_classes=self.y_test.size(1),
                          hidden_sizes=hidden_dims)


    def get_optimizer(self, model) -> torch.optim.Optimizer:
        return torch.optim.NAdam(model.parameters(), lr=0.001)
