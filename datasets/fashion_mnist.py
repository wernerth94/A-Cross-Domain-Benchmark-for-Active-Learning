from typing import Tuple
import torch
import torch.nn as nn
import torchvision
from core.data import BaseDataset, postprocess_torch_dataset, convert_to_channel_first, subsample_data
from core.classifier import ConvolutionalModel

class FashionMnist(BaseDataset):
    def __init__(self, pool_rng, budget=1000, initial_points_per_class=100, classifier_batch_size=64,
                 data_file="fashion_mnist_al.pt",
                 cache_folder:str="~/.al_benchmark/datasets"):
        # TODO: decide on a budget
        super().__init__(budget, initial_points_per_class, classifier_batch_size, data_file, pool_rng, cache_folder)


    def _download_data(self, test_data_fraction=0.1):
        train = torchvision.datasets.FashionMNIST(root=self.cache_folder, train=True, download=True)
        test = torchvision.datasets.FashionMNIST(root=self.cache_folder, train=False, download=True)
        x_train, self.y_train, x_test, y_test = postprocess_torch_dataset(train, test)
        # add an explicit color dimension for compatibility
        x_train = torch.unsqueeze(x_train, -1)
        x_test = torch.unsqueeze(x_test, -1)
        x_test, self.y_test = subsample_data(x_test, y_test, test_data_fraction)
        self.x_train, self.x_test = convert_to_channel_first(x_train, x_test)
        # normalize pixel values from [0..255] to [-1..1]
        high = 255.0
        self.x_train = self.x_train / (high / 2.0) - 1.0
        self.x_test = self.x_test / (high / 2.0) - 1.0
        print("Download successful")


    def get_classifier(self, hidden_dims:Tuple[int]=(12, 24, 48)) -> nn.Module:
        from core.classifier import ConvolutionalModel
        model = ConvolutionalModel(input_size=self.x_shape,
                                   num_classes=self.n_classes,
                                   hidden_sizes=hidden_dims)
        return model


    def get_optimizer(self, model, lr=0.01, weight_decay=0.0) -> torch.optim.Optimizer:
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)


    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: TorchVision\n" \
             "Normalization: Linear between [-1..1]\n" \
             "Classifier: Vanilla ConvNet"
        return s

