
from core.data import BaseDataset, postprocess_torch_dataset, convert_to_channel_first, subsample_data
import numpy as np
from sklearn.model_selection import train_test_split


class SynthData(BaseDataset):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None,):
        fitting_mode = "from_scratch" if encoded else "finetuning"
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, fitting_mode)



    def createToy_3Clust(self, n_perClust=50, cov=[[1, 0], [0, 1]] ):

        self.filename_3Clust = 'toydata_3Clust.npy'
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

        np.save(self.filename_3Clust, np.concatenate((data_pos, data_neg), axis=0))

    def creatToy_Sissor(self, n_samples=10,  n_clusters=10, dist_cluster=2, cov=[[1, 0], [0, 1]] ):

        # Generate data for two classes from 4 clusters
        # n_samples : pro cluster - divide by 2 for n_samples per class
        # n_clusters : Number of Clusters along the decision border
        # dist_cluster : distance of clusters

        self.filename_sissor = 'toydata_sissor.npy'
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

        np.save(self.filename_sissor, np.concatenate((data_pos,data_neg),axis=0))

    def _download_data(self, dataset='3Clust', train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
        assert train_ratio + val_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        if dataset == '3Clust':
            data = self.filename_3Clust
        elif dataset == 'Sissor':
            data = self.filename_sissor

        self.train_data, remaining_data = train_test_split(data, train_size=train_ratio, random_state=42)
        self.val_data, self.test_data = train_test_split(remaining_data, train_size=val_ratio / (val_ratio + test_ratio),
                                               random_state=42)


