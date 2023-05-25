import os, shutil
from os.path import join, exists
import pandas as pd
import yaml

def load_budget(dataset_name:str):
    dataset_name = dataset_name.lower()
    encoded = False
    if dataset_name.endswith("encoded"):
        encoded = True
        dataset_name = dataset_name[:-7]
    with open(f"configs/{dataset_name}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    if encoded:
        return config["dataset_embedded"]["budget"]
    else:
        return config["dataset"]["budget"]


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
]


for dataset_name in datasets:
    dataset_dir = join("runs", dataset_name)
    budget = load_budget(dataset_name)

    if not exists(dataset_dir):
        print(f"{dataset_name} does not exist")
        continue
    relevant_agents = os.listdir(dataset_dir)
    relevant_agents = [r for r in relevant_agents if r not in excluded_agents]
    for agent in relevant_agents:
        agent_dir = join(dataset_dir, agent)
        acc_file = join(agent_dir, "accuracies.csv")
        loss_file = join(agent_dir, "losses.csv")
        for f in [acc_file, loss_file]:
            if exists(f):
                df = pd.read_csv(f, header=0, index_col=0)
                if len(df) > budget:
                    df = df[:budget]
                    df.to_csv(f)
