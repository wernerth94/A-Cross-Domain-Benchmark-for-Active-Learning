import os
from os.path import exists
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from core.data import GaussianNoise, VectorToTensor
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, VectorDataset, normalize, postprocess_svm_data
import requests

class DNA(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="dna_al.pt"):
        self.train_file = os.path.join(cache_folder, "dna_train.txt")
        self.test_file = os.path.join(cache_folder, "dna_test.txt")
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)


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
            self._convert_data_to_tensors()
            print("Download successful")

    def load_pretext_data(self)->tuple[Dataset, Dataset]:
        if exists(self.train_file) and exists(self.test_file):
            train = load_svmlight_file(self.train_file, n_features=180)
            test = load_svmlight_file(self.test_file, n_features=180)
            x_train, y_train, x_test, y_test = postprocess_svm_data(train, test, target_to_one_hot=False)
            x_train, x_test = normalize(x_train, x_test, mode="min_max")
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            train_dataset = VectorDataset(x_train, torch.from_numpy(y_train))
            test_dataset = VectorDataset(x_test, torch.from_numpy(y_test))
            return (train_dataset, test_dataset)

    def get_pretext_transforms(self, config)->transforms.Compose:
        return transforms.Compose([
                VectorToTensor(),
                GaussianNoise(config["pretext_transforms"]["gauss_scale"])
            ])

    def get_pretext_validation_transforms(self, config)->transforms.Compose:
        return transforms.Compose([
                VectorToTensor(),
            ])

    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: LibSVMTools\n" \
             "Normalization: Linear between [0..1]\n" \
             "Classifier: DenseNet"
        return s
