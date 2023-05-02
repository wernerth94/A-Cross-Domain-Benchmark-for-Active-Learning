from typing import Tuple
import numpy as np
import requests
from os.path import join, exists
from collections import abc

# from torch._six import string_classes
import torch.utils.data
import torchvision
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
from sklearn.datasets import load_svmlight_file
from core.data import postprocess_torch_dataset, convert_to_channel_first, postprocess_svm_data, normalize


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

    # elif isinstance(batch[0], string_classes):
    #     return batch

    elif isinstance(batch[0], abc.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], abc.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))


def get_train_dataloader_for_dataset(config, data):
    return torch.utils.data.DataLoader(data, num_workers=8,
            batch_size=config['pretext_training']['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_validation_dataloader_for_dataset(config, data):
    return torch.utils.data.DataLoader(data, num_workers=8,
            batch_size=config['pretext_training']['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


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



