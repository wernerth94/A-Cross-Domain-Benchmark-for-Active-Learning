from abc import ABC, abstractmethod
from typing import Tuple, Literal, Union, Callable
import os
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset
from sklearn.preprocessing import MinMaxScaler
from core.helper_functions import to_torch

class BaseDataset(ABC):

    def __init__(self, budget:int,
                 initial_points_per_class:int,
                 classifier_batch_size:int,
                 data_file:str,
                 cache_folder:str="~/.al_benchmark/datasets",
                 device=None,
                 class_fitting_mode:Literal["from_scratch", "finetuning"]="finetuning"):
        assert isinstance(budget, int) and budget > 0, f"The budget {budget} is invalid"
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.budget = budget
        self.classifier_batch_size = classifier_batch_size
        self.class_fitting_mode = class_fitting_mode
        self.data_file = data_file
        self.cache_folder = cache_folder
        self.initial_points_per_class = initial_points_per_class

        self._load_or_download_data()
        self.n_classes = self.y_test.shape[-1]
        self.x_shape = self.x_unlabeled.shape[1:]
        self.name = str(self.__class__).split('.')[-1][:-2]
        print(f"Loaded dataset: {self.name}")
        print(f"| Number of classes: {self.n_classes}")
        print(f"| Labeled Instances: {len(self.x_labeled)}")
        print(f"| Unlabeled Instances: {len(self.x_unlabeled)}")
        print(f"| Test Instances {len(self.x_test)}")

    @abstractmethod
    def _download_data(self):
        '''
        Downloads the data from web and saves it into self.cache_folder
        '''
        pass

    def _load_data(self)->Union[None, Tuple]:
        '''
        Loads the data from self.cache_folder
        Returns None on failure
        :return: None or tuple(x_train, y_train, x_test, y_test)
        '''
        file = os.path.join(self.cache_folder, self.data_file)
        if os.path.exists(file):
            dataset = torch.load(file)
            return dataset["x_labeled"], dataset["y_labeled"], \
                dataset["x_unlabeled"], dataset["y_unlabeled"], \
                dataset["x_test"], dataset["y_test"]
        return None

    @abstractmethod
    def _normalize_data(self):
        '''
        Applies normalization to the data
        '''
        pass

    @abstractmethod
    def get_classifier(self, hidden_dims:tuple=tuple())->Module:
        '''
        This creates a torch model that serves as a classification model for this dataset
        :return: PyTorch Model
        '''
        pass


    @abstractmethod
    def get_optimizer(self, model:Module, lr:float=0.001, weight_decay:float=0.0)->Optimizer:
        pass


    def _load_or_download_data(self):
        data = self._load_data()
        if data is None:
            print(f"No local copy found in {self.cache_folder}. \nDownloading Data...")
            self._download_data()
            assert hasattr(self, "x_train")
            assert hasattr(self, "y_train")
            assert hasattr(self, "x_test")
            assert hasattr(self, "y_test")
            self._normalize_data()
            self._convert_data_to_tensors()
            self._create_seed_set()
            data = self._load_data()
            if data is None:
                raise ValueError(f"Dataset was not found in {self.cache_folder} and could not be downloaded")
        self.x_labeled, self.y_labeled, self.x_unlabeled, self.y_unlabeled, self.x_test, self.y_test = data
        return True


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

        out_file = os.path.join(self.cache_folder, self.data_file)
        torch.save({
            "x_labeled": self.x_labeled,
            "y_labeled": self.y_labeled,
            "x_unlabeled": self.x_unlabeled,
            "y_unlabeled": self.y_unlabeled,
            "x_test": self.x_test,
            "y_test": self.y_test,
        }, out_file)
        return True


    def _convert_data_to_tensors(self):
        self.x_train = to_torch(self.x_train, torch.float32, device=self.device)
        self.y_train = to_torch(self.y_train, torch.float32, device=self.device)
        self.x_test = to_torch(self.x_test, torch.float32, device=self.device)
        self.y_test = to_torch(self.y_test, torch.float32, device=self.device)


    def to(self, device):
        """
        This mirrors the behavior of tensor.to(device), but without copying the data
        :param device: cuda or cpu
        :return: self
        """
        for attr in dir(self):
            if not attr.startswith('__'):
                value = getattr(self, attr)
                if type(value) == torch.Tensor:
                    setattr(self, attr, value.to(device))
        return self

    def get_meta_data(self)->str:
        return f"{self.name}\n" \
               f"Budget: {self.budget}\n" \
               f"Seeding points per class: {self.initial_points_per_class}\n" \
               f"Classifier Batch Size: {self.classifier_batch_size}"



##################################################################
# Data loading functions, etc.

def normalize(x_train, x_test, mode:Literal["none", "mean", "mean_std", "min_max"]="min_max"):
    if mode == "mean":
        x_train = (x_train - np.mean(x_train, axis=0))
        x_test = (x_test - np.mean(x_test, axis=0))
    elif mode == "mean_std":
        std_train, std_test = np.std(x_train, axis=0), np.std(x_test, axis=0)
        std_train[std_train == 0.0] = 1.0 # replace 0 to avoid division by 0
        std_test[std_test == 0.0] = 1.0
        x_train = (x_train - np.mean(x_train, axis=0)) / std_train
        x_test = (x_test - np.mean(x_test, axis=0)) / std_test
    elif mode == "min_max":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        x_train, x_test = np.nan_to_num(x_train), np.nan_to_num(x_test)
    else:
        raise ValueError(f"Normalization not known: {mode}")
    return x_train, x_test


def postprocess_torch_dataset(train:VisionDataset, test:VisionDataset)->Tuple:
    x_train, y_train = train.data, np.array(train.targets)
    x_test, y_test = test.data, np.array(test.targets)
    one_hot_train = np.zeros((len(y_train), y_train.max() + 1))
    one_hot_train[np.arange(len(y_train)), y_train] = 1
    one_hot_test = np.zeros((len(y_test), y_test.max() + 1))
    one_hot_test[np.arange(len(y_test)), y_test] = 1
    return x_train, one_hot_train, x_test, one_hot_test

def postprocess_svm_data(train:tuple, test:tuple)->Tuple:
    # convert labels to int
    x_train, y_train = train[0], train[1].astype(int)
    x_test, y_test = test[0], test[1].astype(int)
    # convert inputs to numpy arrays
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    # convert svm labels to onehot
    mask = y_train == -1
    y_train[mask] += 1
    mask = y_test == -1
    y_test[mask] += 1
    one_hot_train = np.zeros((len(y_train), y_train.max() + 1))
    one_hot_train[np.arange(len(y_train)), y_train] = 1
    one_hot_test = np.zeros((len(y_test), y_test.max() + 1))
    one_hot_test[np.arange(len(y_test)), y_test] = 1
    return x_train, one_hot_train, x_test, one_hot_test


def load_numpy_dataset(file_name:str)->Union[None, Tuple]:
    if os.path.exists(file_name):
        try:
            with np.load(os.path.join(file_name, file_name), allow_pickle=True) as f:
                x_train, y_train = f['x_train'], f['y_train']
                x_test, y_test = f['x_test'], f['y_test']
            return x_train, y_train, x_test, y_test
        except:
            pass
    return None