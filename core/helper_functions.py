import functools
from typing import Callable
import os
from os.path import join, exists
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class EarlyStopping:
    def __init__(self, patience=7, lower_is_better=True):
        self.patience = patience
        self.lower_is_better = lower_is_better
        self.best_loss = torch.inf if lower_is_better else -torch.inf
        self.steps_without_improvement = 0
    def check_stop(self, loss_val):
        if (self.lower_is_better     and loss_val >= self.best_loss) or \
           (not self.lower_is_better and loss_val <= self.best_loss):
            self.steps_without_improvement += 1
            if self.steps_without_improvement > self.patience:
                return True
        else:
            self.steps_without_improvement = 0
            self.best_loss = loss_val
        return False

def save_meta_data(logpath, agent, env, dataset, config:dict):
    if not os.path.exists(logpath):
        os.makedirs(logpath, exist_ok=True)
    file = os.path.join(logpath, "meta.txt")
    if os.path.exists(file):
        os.remove(file)

    with open(file, "w") as f:
        if hasattr(dataset, "get_meta_data"):
            f.write("# Dataset: \n")
            f.write(f"{dataset.get_meta_data()} \n\n")
        if hasattr(agent, "get_meta_data"):
            f.write("# Agent: \n")
            f.write(f"{agent.get_meta_data()} \n\n")
        if hasattr(env, "get_meta_data"):
            f.write("# Environment: \n")
            f.write(f"{env.get_meta_data()} \n\n")

        f.write("# Config: \n")
        for key, value in config.items():
            f.write(f"{key}: {value} \n")


def _pad_nans_with_last_value(df:pd.DataFrame):
    max_len = len(df)
    for col in df:
        diff = max_len - sum(pd.notna(df[col]))
        if diff > 0:
            last_val = df[col][sum(pd.notna(df[col])) - 1]
            df[col] = pd.concat([df[col].iloc[:-diff], pd.Series([last_val]*diff)], ignore_index=True)
    return df


initial_pool_size = {
    "Splice": 2,
    "SpliceEncoded": 2,
    "DNA": 3,
    "DNAEncoded": 3,
    "USPS": 10,
    "USPSEncoded": 10,
    "Cifar10": 1000,
    "Cifar10Encoded":10,
    "FashionMnist": 1000,
    "FashionMnistEncoded": 10,
    "TopV2": 7,
    "News": 15,
    "Mnist": 10,
    "DivergingSin": 2,
    "ThreeClust": 2
}

def get_init_pool_size(dataset_agent:str):
    dataset = dataset_agent.split("/")[0]
    if dataset not in initial_pool_size:
        print(f"Dataset {dataset} has no initial pool size")
        return 0
    else:
        return initial_pool_size[dataset]

def _moving_avrg(line, weight):
    moving_mean = line[0]
    result = [moving_mean]
    for i in range(1, len(line)):
        moving_mean = weight * moving_mean + (1 - weight) * line[i]
        result.append(moving_mean)
    return np.array(result)


def moving_avrg(trajectory, weight):
    # moving average for a tuple of trajectory and std
    stdCurve = trajectory[1]
    trajectory = trajectory[0]
    return _moving_avrg(trajectory, weight), _moving_avrg(stdCurve, weight)


def plot_upper_bound(dataset, budget, color, alpha=0.8, percentile=0.99, linewidth=2, run_name="UpperBound"):
    file = os.path.join("/home/thorben/phd/projects/al-benchmark/runs", dataset, f"{run_name}/accuracies.csv")
    all_runs = pd.read_csv(file, header=0, index_col=0)
    # mean = np.mean(all_runs.values, axis=1)
    mean = np.median(all_runs.values, axis=1)
    mean_percentile = percentile * mean
    # mean_percentile = percentile * mean
    mean = [float(mean)]*budget
    mean_percentile = [float(mean_percentile)]*budget
    x = np.arange(budget) + get_init_pool_size(dataset)
    plt.plot(x, mean, label="Full Dataset", linewidth=linewidth, c=color, alpha=alpha)
    plt.plot(x, mean_percentile, label="99% Percentile", linewidth=1, linestyle='--', c=color, alpha=0.6)

def plot_benchmark(dataset, color, display_name, smoothing_weight=0.0, alpha=0.8, linewidth=1.5, plot_std=False, show_auc=False):
    full_name = f"{display_name}"
    file = os.path.join("/home/thorben/phd/projects/al-benchmark/runs", dataset, "accuracies.csv")
    all_runs = pd.read_csv(file, header=0, index_col=0)
    if show_auc:
        values = all_runs.values
        auc = np.sum(values, axis=0) / values.shape[0]
        full_name += " - AUC: %1.3f"%(np.median(auc).item())
    # mean = np.median(all_runs.values, axis=1)
    mean = np.mean(all_runs.values, axis=1)
    std = np.std(all_runs.values, axis=1)
    curve = np.stack([mean, std])
    if smoothing_weight > 0.0:
        avrg_curve, std_curve = moving_avrg(curve, smoothing_weight)
    else:
        avrg_curve, std_curve = mean, std
    x = np.arange(len(avrg_curve)) + get_init_pool_size(dataset)
    if plot_std:
        if show_auc:
            avrg_std = round(sum(std) / len(std), 3)
            full_name += f"+-{avrg_std}"
        plt.fill_between(x, avrg_curve-std_curve, avrg_curve+std_curve, alpha=0.5, facecolor=color)
    plt.plot(x, avrg_curve, label=full_name, linewidth=linewidth, c=color, alpha=alpha)
    return len(x)

def plot_batch_benchmark(dataset, color, display_name, alpha=0.8, linewidth=1.5, plot_std=False, show_auc=True):
    full_name = f"{display_name}"
    file = os.path.join("/home/thorben/phd/projects/al-benchmark/runs", dataset, "accuracies.csv")
    all_runs = pd.read_csv(file, header=0, index_col=0)
    all_runs = all_runs.dropna(axis=0)
    if show_auc:
        values = all_runs.values
        auc = np.sum(values, axis=0) / values.shape[0]
        full_name += " - AUC: %1.3f"%(np.median(auc).item())
    x = list(all_runs.index)
    x = [i + get_init_pool_size(dataset) for i in x]
    mean = np.mean(all_runs.values, axis=1)
    # mean = np.median(all_runs.values, axis=1)
    std = np.std(all_runs.values, axis=1)
    if plot_std:
        plt.fill_between(x, mean-std, mean+std, alpha=0.5, facecolor=color)
    plt.plot(x, mean, label=full_name, linewidth=linewidth, c=color, alpha=alpha)
    return len(x)

def plot_learning_curves(list_of_accs:list, out_file:str=None):
    for accs in list_of_accs:
        x = range(len(accs))
        plt.plot(x, accs, alpha=0.7)

    plt.xlabel("epochs")
    plt.ylabel("validation accuracy")
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.clf()

def sort_by_run_id(x, y):
    """
    custom comparator for sorting run folders with syntax 'run_<id>'
    """
    if "_" not in x and "_" not in x:
        return 0
    elif "_" not in x:
        return 1
    elif "_" not in y:
        return -1
    else:
        x_id = int(x.split("_")[-1])
        y_id = int(y.split("_")[-1])
        if x_id > y_id:
            return 1
        elif x_id < y_id:
            return -1
        else:
            return 0


def collect_results(base_path, folder_prefix):
    result_acc = pd.DataFrame()
    result_loss = pd.DataFrame()
    runs = sorted(os.listdir(base_path), key=functools.cmp_to_key(sort_by_run_id))
    for run_folder in runs:
        if run_folder.startswith(folder_prefix):
            acc_file_path = join(base_path, run_folder, "accuracies.csv")
            if exists(acc_file_path):
                accuracies = pd.read_csv(acc_file_path, header=0, index_col=0)
                result_acc = pd.concat([result_acc, accuracies], axis=1, ignore_index=True)

            loss_file_path = join(base_path, run_folder, "losses.csv")
            if exists(loss_file_path):
                losses = pd.read_csv(loss_file_path, header=0, index_col=0)
                result_loss = pd.concat([result_loss, losses], axis=1, ignore_index=True)
    result_acc.to_csv(join(base_path, "accuracies.csv"))
    result_loss.to_csv(join(base_path, "losses.csv"))


def get_dataset_by_name(name:str)->Callable:
    import datasets
    # Tabular
    name = name.lower()
    if name == "splice":
        return datasets.Splice
    elif name == "dna":
        return datasets.DNA
    elif name == "usps":
        return datasets.USPS
    # Image
    elif name == "cifar10":
        return datasets.Cifar10
    elif name == "mnist":
        return datasets.Mnist
    elif name == "fashionmnist":
        return datasets.FashionMnist
    # Text
    elif name == "topv2":
        return datasets.TopV2
    elif name == "news":
        return datasets.News
    # Toy
    elif name == 'threeclust':
        return datasets.ThreeClust
    elif name == 'divergingsin':
        return datasets.DivergingSin

    else:
        raise ValueError(f"Dataset name '{name}' not recognized")


def get_agent_by_name(name:str)->Callable:
    import agents
    name = name.lower()
    if name == "random":
        return agents.RandomAgent
    elif name == "entropy":
        return agents.ShannonEntropy
    elif name == "margin":
        return agents.MarginScore
    elif name == "coreset":
        return agents.Coreset_Greedy
    elif name == "bald":
        return agents.BALD
    elif name == "badge":
        return agents.Badge
    elif name == "typiclust":
        return agents.TypiClust
    elif name == "coregcn":
        return agents.CoreGCN
    elif name == "dsa":
        return agents.DSA
    elif name == "lsa":
        return agents.LSA
    else:
        raise ValueError(f"Agent name '{name}' not recognized")


if __name__ == '__main__':
    # plot_batch_benchmark("Splice/5/Badge", "b", "random")
    base_path = "runs/Splice/50/TypiClust"
    collect_results(base_path, "run_")
