import numpy as np
from core.data import BaseDataset, normalize
from sklearn.model_selection import train_test_split
import torchvision


class SynthData(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='ThreeClust'):
        assert not encoded, "This dataset does not support encodings"
        self.dataset = dataset
        fitting_mode = "from_scratch" if encoded else "finetuning"
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, fitting_mode)



    def createToy_ThreeClust(self, n_perClust=150, cov=[[1, 0], [0, 1]] ):

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

        data_pos = np.c_[data_pos, np.ones(len(data_pos))]
        data_neg = np.c_[data_neg, np.zeros(len(data_neg))]

        return np.concatenate((data_pos, data_neg), axis=0)

    def creatToy_Scissor(self, n_samples=50,  n_clusters=10, dist_cluster=2, cov=[[1, 0], [0, 1]] ):

        # Generate data for two classes from 4 clusters
        # n_samples : pro cluster - divide by 2 for n_samples per class
        # n_clusters : Number of Clusters along the decision border
        # dist_cluster : distance of clusters

        pos_cls = []
        neg_cls = []

        for i in range(n_clusters):
            mean_pos_cls = [i * dist_cluster, 2 + 0.5 * i]
            mean_neg_cls = [i * dist_cluster, -2 - 0.5 * i]


            pos_cluster = self.pool_rng.multivariate_normal(mean_pos_cls, cov, n_samples)
            neg_cluster = self.pool_rng.multivariate_normal(mean_neg_cls, cov, n_samples)

            pos_cls.append(pos_cluster)
            neg_cls.append(neg_cluster)


        data_pos = np.concatenate(pos_cls, axis=0)
        data_neg = np.concatenate(neg_cls, axis=0)
        data_pos = np.c_[data_pos, np.ones(len(data_pos))]
        data_neg = np.c_[data_neg, np.zeros(len(data_neg))]

        return np.concatenate((data_pos,data_neg),axis=0)

    def creatDivergentSin(self, n_samples=100, divergence_factor=0.5, sin_freq=2, cov=0.3):

        x = np.linspace(0, 10, n_samples)
        sin_curve = np.sin(sin_freq*x)

        # Cluster above the curve
        cluster_above_y = sin_curve + divergence_factor * x + self.pool_rng.random.normal(0, cov, n_samples)

        # Cluster below the curve
        cluster_below_y = sin_curve - divergence_factor * x + self.pool_rng.random.normal(0, cov, n_samples)

        data_pos = np.c_[cluster_above_y, np.ones(len(cluster_above_y))]
        data_neg = np.c_[cluster_below_y, np.zeros(len(cluster_below_y))]

        return np.concatenate((data_pos, data_neg), axis=0)

    def _download_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        raise NotImplementedError

    def _load_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        assert train_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        if self.dataset == 'ThreeClust':
            data = self.createToy_ThreeClust()
        elif self.dataset == 'Scissor':
            data = self.creatToy_Scissor()
        elif self.dataset == 'Scissor':
            data = self.creatDivergentSin()
        else:
            raise NotImplementedError

        ids = np.arange(data.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        x_train = data[train_ids, :2]
        y_train = to_one_hot(data[train_ids, -1].astype(int))
        x_test = data[test_ids, :2]
        y_test = to_one_hot(data[test_ids, -1].astype(int))

        x_train, x_test = normalize(x_train, x_test, mode="min_max")
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return to_torch(x_train, torch.float32, device=self.device), \
               to_torch(y_train, torch.float32, device=self.device), \
               to_torch(x_test, torch.float32, device=self.device), \
               to_torch(y_test, torch.float32, device=self.device),


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


class Scissor(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='Scissor'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)

class DivergentSin(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='DivergentSin'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)

