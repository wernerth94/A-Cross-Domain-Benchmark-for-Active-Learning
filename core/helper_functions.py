from typing import Union, Callable
import os
import math
from os.path import join, exists
import torch
from torch.nn.init import calculate_gain, _calculate_correct_fan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from core.agent import BaseAgent
import agents
from core.data import BaseDataset
import datasets

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

def save_meta_data(logpath, agent, env, dataset, additional:dict=None):
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
        if additional is not None:
            f.write("# Other: \n")
            for key, value in additional.items():
                f.write(f"{key}: {value} \n")


def plot_mean_std_development(inpt:list, title:str, out_file:str=None):
    # standard deviation statistics
    mean_develop = [np.mean(inpt[:i]) for i in range(1, len(inpt) + 1)]
    median_develop = [np.median(inpt[:i]) for i in range(1, len(inpt) + 1)]
    std_develop = [np.std(inpt[:i]) for i in range(1, len(inpt) + 1)]
    fig, ax = plt.subplots()
    ax.set_ylabel("mean")
    ax.set_xlabel("eval run")
    ax.grid()
    ax.scatter(range(len(inpt)), inpt, c="r", s=9, alpha=0.3)
    ax.plot(mean_develop, c="b", label="mean")
    ax.plot(median_develop, c="navy", label="median")
    ax2 = ax.twinx()
    ax2.set_ylabel("std")
    ax2.plot(std_develop, c="g", label="std")

    legend_elements = [
        Patch(facecolor="b", label="mean"),
        Patch(facecolor="navy", label="median"),
        Patch(facecolor="g", label="std"),
    ]
    ax2.legend(handles=legend_elements)
    ax.set_title(title)
    if out_file is None:
        plt.show()
    else:
        fig.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.close(fig)


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


def collect_results(base_path, folder_prefix):
    def check_for_nan_cols(df: pd.DataFrame):
        cleaned_pd = df.copy(deep=True)
        for col_name in df:
            if df[col_name].isnull().values.any():
                cleaned_pd.drop(columns=[col_name], inplace=True)
        return cleaned_pd

    result_acc = pd.DataFrame()
    result_loss = pd.DataFrame()
    for run_folder in os.listdir(base_path):
        if run_folder.startswith(folder_prefix):
            acc_file_path = join(base_path, run_folder, "accuracies.csv")
            if exists(acc_file_path):
                accuracies = pd.read_csv(acc_file_path, header=0, index_col=0)
                accuracies = check_for_nan_cols(accuracies)
                result_acc = pd.concat([result_acc, accuracies], axis=1, ignore_index=True)

            loss_file_path = join(base_path, run_folder, "losses.csv")
            if exists(loss_file_path):
                losses = pd.read_csv(loss_file_path, header=0, index_col=0)
                losses = check_for_nan_cols(losses)
                result_loss = pd.concat([result_loss, losses], axis=1, ignore_index=True)
    result_acc.to_csv(join(base_path, "accuracies.csv"))
    result_loss.to_csv(join(base_path, "losses.csv"))


def get_dataset_by_name(name:str)->Union[Callable, BaseDataset]:
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
    elif name == "fashion_mnist":
        return datasets.FashionMnist
    # Toy
    elif name == 'toydata':
        return datasets.ToyData

    else:
        raise ValueError(f"Dataset name '{name}' not recognized")


def get_agent_by_name(name:str)->Union[Callable, BaseAgent]:
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
    elif name == "sal":
        return agents.SAL
    else:
        raise ValueError(f"Agent name '{name}' not recognized")
