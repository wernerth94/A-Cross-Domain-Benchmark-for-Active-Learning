from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from core.resnet import ResNet18
from sim_clr.encoder import ContrastiveModel
from core.data import BaseDataset, postprocess_torch_dataset, convert_to_channel_first, subsample_data

class FashionMnist(BaseDataset):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="fashion_mnist_al.pt",):
        # TODO: decide on a budget
        fitting_mode = "from_scratch" if encoded else "finetuning"
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, fitting_mode)


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


    def load_pretext_data(self)->tuple[Dataset, Dataset]:
        train_dataset = torchvision.datasets.FashionMNIST(root=self.cache_folder, train=True, download=True)
        val_dataset = torchvision.datasets.FashionMNIST(root=self.cache_folder, train=False, download=True)
        train_dataset.targets = torch.Tensor(train_dataset.targets).int()
        val_dataset.targets = torch.Tensor(val_dataset.targets).int()
        return (train_dataset, val_dataset)

    def get_pretext_transforms(self, config:dict)->transforms.Compose:
        return transforms.Compose([
                transforms.RandomResizedCrop(size=28),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([
                #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                #                            hue=0.1)
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

    def get_pretext_validation_transforms(self, config:dict)->transforms.Compose:
        return transforms.Compose([
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: TorchVision\n" \
             "Normalization: Linear between [-1..1]\n" \
             "Classifier: ResNet18"
        return s

