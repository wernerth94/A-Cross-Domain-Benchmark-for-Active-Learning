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



for dataset_name in datasets:
    dataset_dir = join("runs", dataset_name)

    if not exists(dataset_dir):
        print(f"{dataset_name} does not exist")
        continue
    relevant_agents = os.listdir(dataset_dir)
    relevant_agents = [r for r in relevant_agents if r not in excluded_agents]
    for agent in relevant_agents:
        agent_dir = join(dataset_dir, agent)
        collect_results(agent_dir, "run_")
