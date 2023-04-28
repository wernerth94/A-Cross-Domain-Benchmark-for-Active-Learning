from typing import Tuple, Union
import numpy as np
from core.data import BaseDataset


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
