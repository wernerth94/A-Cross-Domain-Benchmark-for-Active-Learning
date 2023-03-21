import numpy as np
from collections import abc
from torch._six import string_classes
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from core.data import postprocess_torch_dataset, convert_to_channel_first

def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], abc.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], abc.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))


def get_train_dataloader_for_dataset(dataset_name, data):
    if dataset_name == "cifar10":
        return torch.utils.data.DataLoader(data, num_workers=8,
                batch_size=512, pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=True)
    else:
        raise NotImplementedError


def get_validation_dataloader_for_dataset(dataset_name, data):
    if dataset_name == "cifar10":
        return torch.utils.data.DataLoader(data, num_workers=8,
                batch_size=512, pin_memory=True, collate_fn=collate_custom,
                drop_last=False, shuffle=False)
    else:
        raise NotImplementedError


class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        self.target_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample[0]
        new_sample = {
            'image': self.image_transform(image),
            'image_augmented': self.augmentation_transform(image),
            'target': sample[1]
        }
        return new_sample
        # sample['image'] = self.image_transform(image)
        # sample['image_augmented'] = self.augmentation_transform(image)
        # return sample


def get_raw_data_by_name(cache_folder, name:str)->tuple:
    if name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=cache_folder, train=True, download=True)
        val_dataset = torchvision.datasets.CIFAR10(root=cache_folder, train=False, download=True)
        y_train = np.array(train_dataset.targets)
        one_hot_train = np.zeros((len(y_train), y_train.max() + 1))
        one_hot_train[np.arange(len(y_train)), y_train] = 1
        train_dataset.targets = torch.from_numpy(y_train)

        y_test = np.array(val_dataset.targets)
        one_hot_test = np.zeros((len(y_test), y_test.max() + 1))
        one_hot_test[np.arange(len(y_test)), y_test] = 1
        val_dataset.targets = torch.from_numpy(y_test)
        return (train_dataset, val_dataset)
    else:
        raise NotImplementedError


def get_transforms_for_dataset(dataset:str):
    if dataset == "cifar10":
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
    else:
        raise NotImplementedError


def get_validation_transforms_for_dataset(dataset:str):
    if dataset == "cifar10":
        return transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
    else:
        raise NotImplementedError
