import numpy as np
from core.data import BaseDataset, normalize
from sklearn.model_selection import train_test_split


class SynthData(BaseDataset):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='3Clust'):
        self.dataset = dataset
        fitting_mode = "from_scratch" if encoded else "finetuning"
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, fitting_mode)



    def createToy_3Clust(self, n_perClust=50, cov=[[1, 0], [0, 1]] ):

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

        self.data_3Clust = np.concatenate((data_pos, data_neg), axis=0)

    def creatToy_Sissor(self, n_samples=10,  n_clusters=10, dist_cluster=2, cov=[[1, 0], [0, 1]] ):

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

        self.data_Sissor = np.concatenate((data_pos,data_neg),axis=0)

    def _download_data(self, dataset='3Clust', train_ratio=0.8, test_ratio=0.20):
        assert train_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        if self.dataset == '3Clust':
            data = self.data_3Clust
        elif self.dataset == 'Sissor':
            data = self.data_Sissor

        self.x_train = data[:, 0: 2]
        self.y_train = data[:,-1]
        self.x_test  = data[:, 0: 2]
        self.y_test  = data[:,-1]


        ids = np.arange(self.x_train.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        self.x_train = self.x_train[train_ids]
        self.y_train = self.y_train[train_ids]
        self.x_test = self.x_train[test_ids]
        self.y_test = self.y_train[test_ids]


        self.x_train, self.x_test = normalize(self.x_train, self.x_test, mode="min_max")
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        self._convert_data_to_tensors()


