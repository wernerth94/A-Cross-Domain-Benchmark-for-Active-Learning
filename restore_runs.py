import os, shutil
from os.path import join, exists
import pandas as pd
from core.helper_functions import collect_results
import yaml

excluded_agents = [
    "UpperBound"
]

datasets = [
    "DNA",
    "DNAEncoded",
    "Splice",
    "SpliceEncoded",
    "USPS",
    "USPSEncoded",
    "Cifar10",
    "Cifar10Encoded",
    "Mnist",
    "MnistEncoded",
    "FashionMnist",
    "FashionMnistEncoded",
    "News",
    "TopV2",
    "DivergingSin",
    "ThreeClust"
]


# fix fuckup with oracles that have num_datapoints_added > 1
# for dataset_name in ["Cifar10", "FashionMnist"]:
#     agent_dir = join("runs", dataset_name, "Oracle")
#     if not exists(agent_dir):
#         continue
#     for file in os.listdir(agent_dir):
#         if file.startswith("run_"):
#             run_folder = join(agent_dir, file)
#             acc_file_path = join(run_folder, "accuracies.csv")
#             if exists(acc_file_path):
#                 accuracies = pd.read_csv(acc_file_path, header=0, index_col=0)
#                 corrected_values = []
#                 for i, series in accuracies.iterrows():
#                     corrected_values.append(series.values[0])
#                     corrected_values.append(pd.NA)
#                 new_df = pd.DataFrame(corrected_values)
#                 new_df.to_csv(acc_file_path)
#
#             loss_file_path = join(run_folder, "losses.csv")
#             if exists(loss_file_path):
#                 losses = pd.read_csv(loss_file_path, header=0, index_col=0)
#                 corrected_values = []
#                 for i, series in losses.iterrows():
#                     corrected_values.append(series.values[0])
#                     corrected_values.append(pd.NA)
#                 new_df = pd.DataFrame(corrected_values)
#                 new_df.to_csv(loss_file_path)
#
#     collect_results(agent_dir, "run_")



# for dataset_name in datasets:
#     dataset_dir = join("runs", dataset_name)
#
#     if not exists(dataset_dir):
#         print(f"{dataset_name} does not exist")
#         continue
#     relevant_agents = os.listdir(dataset_dir)
#     relevant_agents = [r for r in relevant_agents if r not in excluded_agents]
#     for agent in relevant_agents:
#         agent_dir = join(dataset_dir, agent)
#         collect_results(agent_dir, "run_")
