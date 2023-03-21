from typing import Tuple
import torch
import torch.nn as nn
import torchvision
from core.data import BaseDataset, postprocess_torch_dataset, convert_to_channel_first, subsample_data
from core.classifier import ConvolutionalModel

class Cifar10(BaseDataset):
    def __init__(self, pool_rng, budget=2000, initial_points_per_class=1, classifier_batch_size=64,
                 data_file="cifar10_al.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, pool_rng, cache_folder)


    def _download_data(self, test_data_fraction=0.1):
        train = torchvision.datasets.CIFAR10(root=self.cache_folder, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=self.cache_folder, train=False, download=True)
        x_train, self.y_train, x_test, y_test = postprocess_torch_dataset(train, test)
        x_test, self.y_test = subsample_data(x_test, y_test, test_data_fraction, self.pool_rng)
        self.x_train, self.x_test = convert_to_channel_first(x_train, x_test)
        # normalize pixel values from [0..255] to [-1..1]
        high = 255.0
        self.x_train = self.x_train / (high / 2.0) - 1.0
        self.x_test = self.x_test / (high / 2.0) - 1.0
        print("Download successful")


    def get_classifier(self, model_rng, hidden_dims:Tuple[int]=(24, 12)) -> nn.Module:
        from core.resnet import ResNet18
        model = ResNet18()
        # model = ConvolutionalModel(input_size=self.x_shape,
        #                            num_classes=self.n_classes,
        #                            hidden_sizes=hidden_dims)
        return model


    def get_optimizer(self, model, lr=0.01, weight_decay=0.0) -> torch.optim.Optimizer:
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)


    def get_sim_clr_config(self) ->dict:
        return {
            # Loss
            "criterion": "simclr",
            "criterion_kwargs": {
               "temperature": 0.1
            },
            # Hyperparameters
            "epochs": 500,
            "optimizer": "sgd",
            "optimizer_kwargs":{
                "nesterov": False,
                 "weight_decay": 0.0001,
                 "momentum": 0.9,
                 "lr": 0.4,
            },
            "scheduler": "cosine",
            "scheduler_kwargs":{
               "lr_decay_rate": 0.1
            },
            "batch_size": 512,
            "num_workers": 8,
            # Transformations
            "augmentation_kwargs":{
                "scale": [0.2, 1.0],

            },
            "transformation_kwargs":{
               "crop_size": 32,
               "normalize":{
                  "mean": [0.4914, 0.4822, 0.4465],
                  "std": [0.2023, 0.1994, 0.2010]
               }
            }
        }

    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: TorchVision\n" \
             "Normalization: Linear between [-1..1]\n" \
             "Classifier: ResNet18"
        return s

