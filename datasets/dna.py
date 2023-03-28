from typing import Tuple, Union, Callable
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, VectorDataset, normalize, postprocess_svm_data
from core.classifier import DenseModel
from sim_clr.encoder import ContrastiveModel
import requests

class DNA(BaseDataset):

    def __init__(self, pool_rng, budget=600, initial_points_per_class=1, classifier_batch_size=64,
                 data_file="dna_al.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        self.train_file = os.path.join(cache_folder, "dna_train.txt")
        self.test_file = os.path.join(cache_folder, "dna_test.txt")
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, pool_rng, cache_folder)


    def _download_data(self, target_to_one_hot=True):
        train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.tr"
        val_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.val"
        test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.t"


        if not exists(self.train_file):
            with open(self.train_file, 'w') as f:
                r = requests.get(train_url)
                f.writelines(r.content.decode("utf-8"))
        if not exists(self.test_file):
            with open(self.test_file, 'w') as f:
                r = requests.get(test_url)
                f.writelines(r.content.decode("utf-8"))

        if exists(self.train_file) and exists(self.test_file):
            train = load_svmlight_file(self.train_file, n_features=180)
            test = load_svmlight_file(self.test_file, n_features=180)
            self.x_train, self.y_train, self.x_test, self.y_test = postprocess_svm_data(train, test,
                                                                                        target_to_one_hot=target_to_one_hot)
            self.x_train, self.x_test = normalize(self.x_train, self.x_test, mode="min_max")
            print("Download successful")

    def load_pretext_data(self)->tuple[Dataset, Dataset]:
        if exists(self.train_file) and exists(self.test_file):
            train = load_svmlight_file(self.train_file, n_features=180)
            test = load_svmlight_file(self.test_file, n_features=180)
            x_train, y_train, x_test, y_test = postprocess_svm_data(train, test, target_to_one_hot=False)
            x_train, x_test = normalize(x_train, x_test, mode="min_max")
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            # we need to emulate a vision dataset for compatibility
            train_dataset = VectorDataset(x_train.reshape([-1, 180, 1, 1]), torch.from_numpy(y_train))
            test_dataset = VectorDataset(x_test.reshape([-1, 180, 1, 1]), torch.from_numpy(y_test))
            return (train_dataset, test_dataset)

    def get_pretext_transforms(self)->transforms.Compose:
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=1),
            ])

    def get_pretext_validation_transforms(self)->transforms.Compose:
        return transforms.Compose([
                transforms.ToTensor(),
            ])

    def get_pretext_encoder(self, config:dict, seed=1) -> nn.Module:
        model_rng = torch.Generator()
        model_rng.manual_seed(seed)
        backbone = DenseModel(model_rng, input_size=60, num_classes=self.n_classes,
                              hidden_sizes=config["encoder"]["encoder_hidden"], add_head=False)
        model = ContrastiveModel({'backbone': backbone, 'dim': config["encoder"]["encoder_hidden"][-1]},
                                 head="linear", features_dim=config["encoder"]["feature_dim"])
        return model

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
