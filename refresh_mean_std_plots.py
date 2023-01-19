from core.helper_functions import plot_mean_std_development
import pandas as pd
import os

for dataset in os.listdir("runs"):
    dataset_folder = os.path.join("runs", dataset)
    for agent in os.listdir(dataset_folder):
        agent_folder = os.path.join(dataset_folder, agent)
        accuracies = pd.read_csv(os.path.join(agent_folder, "accuracies.csv"), header=0, index_col=0)
        values = accuracies.values
        plot_mean_std_development(values[-1, :] - values[0, :],
                                  "Improvement",
                                  os.path.join(agent_folder, "mean_std_convergence_improvement.jpg"))
        plot_mean_std_development(values[-1, :],
                                  "Final Accuracy",
                                  os.path.join(agent_folder, "mean_std_convergence_final_value.jpg"))
