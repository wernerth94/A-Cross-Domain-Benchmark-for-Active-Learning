from typing import Tuple, Union, Callable
import os
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from core.data import BaseDataset, normalize, postprocess_svm_data
from core.classifier import DenseModel
import requests

class ToyData1(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng):
        super().__init__(cache_folder, config, pool_rng,
                         encoded=False, data_file="", class_fitting_mode="from_scratch")


    def _download_data(self, target_to_one_hot=True):
        self.x_train = np.zeros(3)
        self.x_test  = np.zeros(3)
        self.y_train = np.zeros((3,2))
        self.y_test  = np.zeros((3,2))

    def _load_data(self, encoded) -> Union[None, Tuple]:
        assert not encoded
        return


    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        return s
