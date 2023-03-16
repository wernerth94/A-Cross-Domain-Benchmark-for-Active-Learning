import numpy as np
import matplotlib.pyplot as plt

# Generate data for two classes from 4 clusters
n_samples = 50 # pro cluster - divide by 2 for n_samples per class
mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples)

mean2 = [4, 3]
cov2 = [[1, 0], [0, 1]]
cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples)

mean3 = [0, 6]
cov3 = [[1, 0], [0, 1]]
cluster3 = np.random.multivariate_normal(mean3, cov3, n_samples)

mean4 = [4, 3]
cov4 = [[1, 0], [0, 1]]
cluster4 = np.random.multivariate_normal(mean4, cov4, n_samples)


data_pos = np.concatenate((cluster1, cluster2), axis=0)
data_neg = np.concatenate((cluster3, cluster4), axis=0)
# Plot the clusters
fig, ax = plt.subplots()
ax.scatter(data_pos[:, 0], data_pos[:, 1], label='Cluster 1')
ax.scatter(data_neg[:, 0], data_neg[:, 1], label='Cluster 2')

ax.legend()
plt.show()

data_pos = np.c_[data_pos, np.ones(len(data_pos))]
data_neg = np.c_[data_neg, np.zeros(len(data_neg))]

np.save('toydata.npy', np.concatenate((data_pos,data_neg),axis=0))




