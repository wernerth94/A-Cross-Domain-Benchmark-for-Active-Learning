from typing import Tuple, Union, Callable
import os
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, normalize, postprocess_svm_data
from core.classifier import DenseModel
import requests
import bz2

class USPS(BaseDataset):
    def __init__(self, pool_rng, budget=900, initial_points_per_class=1, classifier_batch_size=43,
                 data_file="usps_al.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, pool_rng, cache_folder)


    def _download_data(self):

        train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"
        # val_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.val"
        test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"

        train_file = os.path.join(self.cache_folder, "usps_train.txt")
        test_file = os.path.join(self.cache_folder, "usps_test.txt")
        if not exists(train_file):
            with open(train_file, 'w') as f:
                r = requests.get(train_url)
                decompressed = bz2.decompress(r.content)
                f.writelines(decompressed.decode("utf-8"))
        if not exists(test_file):
            with open(test_file, 'w') as f:
                r = requests.get(test_url)
                decompressed = bz2.decompress(r.content)
                f.writelines(decompressed.decode("utf-8"))

        if exists(train_file) and exists(test_file):
            # load the data
            train = load_svmlight_file(train_file, n_features=256)
            test = load_svmlight_file(test_file, n_features=256)
            self.x_train, self.y_train, self.x_test, self.y_test = postprocess_svm_data(train, test)
            self.x_train, self.x_test = normalize(self.x_train, self.x_test, mode="min_max")
            print("Download successful")


    def get_classifier(self, model_rng, hidden_dims :Tuple[int] =(24, 12)) -> nn.Module:
        model = DenseModel(model_rng,
                           input_size=self.x_shape[-1],
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
