import os
import pandas as pd
import numpy as np

for dataset in os.listdir("runs"):
    dataset_folder = os.path.join("runs", dataset)
    leaderboard = pd.DataFrame(columns=["Agent", "Acc", "Improvement", "AUC"])

    agent_idx = 0
    for agent in os.listdir(dataset_folder):
        agent_folder = os.path.join(dataset_folder, agent)
        acc_file = os.path.join(agent_folder, "accuracies.csv")
        if os.path.exists(acc_file):
            accuracies = pd.read_csv(acc_file, header=0, index_col=0)
            values = accuracies.values
            final_acc = values[-1, :]
            final_acc = (np.mean(final_acc), np.std(final_acc))
            improvement = values[-1, :] - values[0, :]
            improvement = (np.mean(improvement), np.std(improvement))
            auc = np.sum(values, axis=0) / values.shape[0]
            auc = (np.mean(auc), np.std(auc))
            leaderboard = pd.concat([
                leaderboard,
                pd.DataFrame({
                    "Agent": agent,
                    "Acc": "%1.4f +- %1.4f"%final_acc,
                    "Improvement": "%1.4f +- %1.4f"%improvement,
                    "AUC": "%1.4f +- %1.4f"%auc,
                }, index=[agent_idx])
            ])
            agent_idx += 1

    if len(leaderboard) > 0:
        leaderboard_file = os.path.join(dataset_folder, "leaderboard.csv")
        if os.path.exists(leaderboard_file):
            os.remove(leaderboard_file)
        leaderboard.to_csv(leaderboard_file)