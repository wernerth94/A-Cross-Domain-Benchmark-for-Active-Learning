from typing import Tuple
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
from torchvision import transforms
from core.resnet import ResNet18
from sim_clr.encoder import ContrastiveModel
from core.data import BaseDataset, postprocess_torch_dataset, convert_to_channel_first, subsample_data

class Cifar10(BaseDataset):
    def __init__(self, pool_rng, encoded,
                 data_file="cifar10_al.pt",
                 pretext_config_file="configs/cifar10.yaml",
                 encoder_model_checkpoint="encoder_checkpoints/cifar10_27.03/model_seed1.pth.tar",
                 budget=200, initial_points_per_class=1, classifier_batch_size=32,
                 cache_folder:str="~/.al_benchmark/datasets"):
        fitting_mode = "from_scratch" if encoded else "finetuning"
        super().__init__(budget, initial_points_per_class, classifier_batch_size,
                         data_file, pretext_config_file, encoder_model_checkpoint,
                         pool_rng, encoded, cache_folder, fitting_mode)


    def _download_data(self, target_to_one_hot=True, test_data_fraction=0.1):
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

    def load_pretext_data(self)->tuple[Dataset, Dataset]:
        train_dataset = torchvision.datasets.CIFAR10(root=self.cache_folder, train=True, download=True)
        val_dataset = torchvision.datasets.CIFAR10(root=self.cache_folder, train=False, download=True)
        train_dataset.targets = torch.Tensor(train_dataset.targets).int()
        val_dataset.targets = torch.Tensor(val_dataset.targets).int()
        return (train_dataset, val_dataset)

    def get_pretext_transforms(self, config:dict)->transforms.Compose:
        return transforms.Compose([
                transforms.RandomResizedCrop(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                           hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

    def get_pretext_validation_transforms(self, config:dict)->transforms.Compose:
        return transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

    def get_pretext_encoder(self, config:dict, seed=None) -> nn.Module:
        backbone = ResNet18(add_head=False)
        model = ContrastiveModel({'backbone': backbone, 'dim':config["encoder"]["encoder_dim"]},
                                 head="mlp", features_dim=config["encoder"]["feature_dim"])
        return model


    def get_classifier(self, model_rng) -> nn.Module:
        if self.encoded:
            model = nn.Sequential(nn.Linear(self.x_shape[-1], self.n_classes))
        else:
            model = ResNet18(num_classes=self.n_classes)
        return model


    def get_optimizer(self, model, lr=0.01, weight_decay=0.0) -> torch.optim.Optimizer:
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)


    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: TorchVision\n" \
             "Normalization: Linear between [-1..1]\n" \
             "Classifier: ResNet18"
        return s

