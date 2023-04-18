from typing import Tuple, Union, Callable
import os
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from core.data import BaseDataset, normalize, postprocess_svm_data
from core.classifier import DenseModel
import requests

class ToyData(BaseDataset):

    def __init__(self, budget=90, initial_points_per_class=1, classifier_batch_size=10,
                 data_file="toydata.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, cache_folder)


    def _download_data(self):
        data = np.load('/home/burchert/reserach/AL/al-benchmark/datasets/toydata.npy')
        train, test = train_test_split(data, test_size= 0.50, random_state=42, shuffle=True)

        self.x_train = train[:,:2]
        self.x_test  = train[:,:2]
        self.y_train = np.asarray(pd.get_dummies(test[:,-1]))
        self.y_test  = np.asarray(pd.get_dummies(test[:,-1]))

    def get_classifier(self, hidden_dims :Tuple[int] =(12, 6)) -> nn.Module:
        model = DenseModel(input_size=self.x_shape[-1],
                           num_classes=self.n_classes,
                           hidden_sizes=hidden_dims)
        return model



    def get_optimizer(self, model, lr=0.001, weight_decay=0.0) -> torch.optim.Optimizer:
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: LibSVMTools\n" \
             "Normalization: Linear between [0..1]\n" \
             "Classifier: DenseNet"
        return s