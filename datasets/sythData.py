import numpy as np
import torch
from core.data import BaseDataset, normalize, to_torch, to_one_hot
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Dataset


class SynthData(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='ThreeClust'):
        assert not encoded, "This dataset does not support encodings"
        self.dataset = dataset
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)


    def create_large_moons(self, n_samples:int=1000, test_ratio=0.8):
        train_samples = int(n_samples * (1.0 - test_ratio))
        test_samples = int(n_samples * test_ratio)

        x_samples = self.pool_rng.uniform(0, 1, train_samples)
        y_samples_pos = self.pool_rng.beta(5.0, 1.0, int(train_samples/2))
        y_samples_neg = 1.0 - self.pool_rng.beta(5.0, 1.0, int(train_samples/2))
        y_samples = np.concatenate([y_samples_pos, y_samples_neg])
        labels = y_samples > (0.1 * np.sin(10.0 * x_samples)) + 0.5
        labels = labels.astype(float)
        x_train = np.stack((x_samples, y_samples), axis=1)
        y_train = labels.copy()

        x_samples = self.pool_rng.uniform(0, 1, test_samples)
        y_samples = self.pool_rng.normal(0.5, 0.1, test_samples)
        labels = y_samples > (0.1 * np.sin(10.0 * x_samples)) + 0.5
        labels = labels.astype(float)
        x_test = np.stack((x_samples, y_samples), axis=1)
        y_test = labels.copy()

        return x_train, y_train, x_test, y_test


    def createToy_ThreeClust(self, n_perClust=150, test_ratio=0.8, cov=[[1, 0], [0, 1]] ):

        mean1 = [0, 0]
        cluster1 = self.pool_rng.multivariate_normal(mean1, cov, n_perClust)

        mean2 = [4, 3]
        cluster2 = self.pool_rng.multivariate_normal(mean2, cov, n_perClust)

        mean3 = [0, 6]
        cluster3 = self.pool_rng.multivariate_normal(mean3, cov, n_perClust)

        mean4 = [4, 3]
        cluster4 = self.pool_rng.multivariate_normal(mean4, cov, n_perClust)

        data_pos = np.concatenate((cluster1, cluster2), axis=0)
        data_neg = np.concatenate((cluster3, cluster4), axis=0)

        x = np.concatenate((data_pos, data_neg), axis=0)
        y = np.concatenate((np.ones(len(data_pos)), np.zeros(len(data_neg))), axis=0)

        ids = np.arange(len(x), dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        x_train = x[train_ids]
        y_train = y[train_ids]
        x_test = x[test_ids]
        y_test = y[test_ids]

        return x_train, y_train, x_test, y_test


    def createDivergingSin(self, n_samples=1000, test_ratio=0.8, divergence_factor=0.5, sin_freq=2, cov=0.3):

        x = np.linspace(0, 10, n_samples)
        sin_curve = np.sin(sin_freq*x)

        # Cluster above the curve
        cluster_above_x = x
        cluster_above_y = sin_curve + divergence_factor * x + self.pool_rng.normal(0, cov, n_samples)
        cluster_above = np.c_[cluster_above_x, cluster_above_y]

        # Cluster below the curve
        cluster_below_x = x
        cluster_below_y = sin_curve - divergence_factor * x + self.pool_rng.normal(0, cov, n_samples)
        cluster_below = np.c_[cluster_below_x, cluster_below_y]

        x = np.concatenate((cluster_above, cluster_below), axis=0)
        y = np.concatenate((np.ones(len(cluster_above_y)), np.zeros(len(cluster_below_y))), axis=0)

        ids = np.arange(len(x), dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        x_train = x[train_ids]
        y_train = y[train_ids]
        x_test = x[test_ids]
        y_test = y[test_ids]

        return x_train, y_train, x_test, y_test


    def _load_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        assert train_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        if self.dataset == 'ThreeClust':
            x_train, y_train, x_test, y_test = self.createToy_ThreeClust(test_ratio=test_ratio)
        elif self.dataset == 'DivergingSin':
            x_train, y_train, x_test, y_test = self.createDivergingSin(test_ratio=test_ratio)
        elif self.dataset == 'LargeMoons':
            x_train, y_train, x_test, y_test = self.create_large_moons(test_ratio=test_ratio)
        else:
            raise NotImplementedError

        y_train = to_one_hot(y_train.astype(int))
        y_test = to_one_hot(y_test.astype(int))

        x_train, x_test = normalize(x_train, x_test, mode="min_max")
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return to_torch(x_train, torch.float32, device=self.device), \
               to_torch(y_train, torch.float32, device=self.device), \
               to_torch(x_test, torch.float32, device=self.device), \
               to_torch(y_test, torch.float32, device=self.device),




    def _download_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        raise NotImplementedError


    # Abstract methods from the Parent
    def get_pretext_transforms(self, config: dict) -> torchvision.transforms.Compose:
        raise NotImplementedError

    def get_pretext_validation_transforms(self, config: dict) -> torchvision.transforms.Compose:
        raise NotImplementedError

    def load_pretext_data(self) -> tuple[Dataset, Dataset]:
        raise NotImplementedError


class ThreeClust(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='ThreeClust'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)

class LargeMoons(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='LargeMoons'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)


class DivergingSin(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='DivergingSin'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)


if __name__ == '__main__':
    import yaml
    with open(f"../configs/divergingsin.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    pool_rng = np.random.default_rng(1)
    dataset = DivergingSin("", config, pool_rng, False)
    import matplotlib.pyplot as plt

    def plot_toy(X, y):
        y = y.cpu()
        X = X.cpu()
        class_indices = np.argmax(y, axis=1)
        matched_data = [(X[i], class_indices[i]) for i in range(len(X))]

        pos_cls = []
        neg_cls = []

        for data in matched_data:
            if data[1] == 0:
                pos_cls.append(data[0])
            else:
                neg_cls.append(data[0])

        pos_x = [tensor[0].item() for tensor in pos_cls]
        pos_y = [tensor[1].item() for tensor in pos_cls]

        neg_x = [tensor[0].item() for tensor in neg_cls]
        neg_y = [tensor[1].item() for tensor in neg_cls]

        fig, ax = plt.subplots()
        ax.scatter(pos_x, pos_y, s=2, label='Class A')
        ax.scatter(neg_x, neg_y, s=2, label='Class B')
        plt.show()


    plot_toy(dataset.x_train, dataset.y_train)
    plot_toy(dataset.x_test, dataset.y_test)
    # y = torch.cat((dataset.y_train, dataset.y_val, dataset.y_test), dim=0)
    # X = torch.cat((dataset.x_train, dataset.x_val, dataset.x_test), dim=0)
    # plot_toy(X, y)
