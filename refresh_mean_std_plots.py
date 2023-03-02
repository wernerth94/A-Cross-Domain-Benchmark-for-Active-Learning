from core.helper_functions import plot_mean_std_development
import pandas as pd
import numpy as np
import os

for dataset in os.listdir("runs"):
    dataset_folder = os.path.join("runs", dataset)
    for agent in os.listdir(dataset_folder):
        agent_folder = os.path.join(dataset_folder, agent)
        acc_file = os.path.join(agent_folder, "accuracies.csv")
        if os.path.exists(acc_file):
            accuracies = pd.read_csv(acc_file, header=0, index_col=0)
            values = accuracies.values
            auc = np.sum(values, axis=0) / values.shape[0]
            try:
                plot_mean_std_development(list(np.squeeze(auc)),
                                          "AUC",
                                          os.path.join(agent_folder, "mean_std_convergence_auc.jpg"))
            except Exception as ex:
                print(f"failed for {acc_file} with exception {str(ex)}")
