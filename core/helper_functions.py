import copy
import functools
import math
from typing import Callable
import os
from os.path import join, exists
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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


def plot_upper_bound(ax, dataset, x_values, color, alpha=0.8, percentile=0.99, linewidth=2, run_name="UpperBound"):
    file = os.path.join("/home/thorben/phd/projects/al-benchmark/runs", dataset, f"{run_name}/accuracies.csv")
    all_runs = pd.read_csv(file, header=0, index_col=0)
    mean = np.mean(all_runs.values, axis=1)
    # mean = np.median(all_runs.values, axis=1)
    mean_percentile = percentile * mean
    mean = [float(mean)]*2
    mean_percentile = [float(mean_percentile)]*2
    x = [min(x_values), max(x_values)]
    ax.plot(x, mean, label="Full Dataset", linewidth=linewidth, c=color, alpha=alpha)
    ax.plot(x, mean_percentile, label="99% Percentile", linewidth=1, linestyle='--', c=color, alpha=0.6)
    return np.mean(all_runs.values, axis=1)


_all_query_sizes = [1, 5, 20, 50, 100, 500]
_all_agents = ["Badge", "BALD", "CoreGCN", "Coreset_Greedy", "DSA", "LSA", "MarginScore", "RandomAgent",
               "ShannonEntropy", "TypiClust", "Oracle"]
_agent_colors = {
    "Oracle": "red",
    "UpperBound": "black",
    "Badge": "orange",
    "BALD": "y",
    "CoreGCN": "violet",
    "Coreset_Greedy": "purple",
    "DSA": "olive",
    "LSA": "brown",
    "MarginScore": "blue",
    "RandomAgent": "grey",
    "ShannonEntropy": "green",
    "TypiClust": "pink"
}
_agent_names = { # only corrected names
    "Coreset_Greedy": "Coreset",
    "MarginScore": "Margin",
    "RandomAgent": "Random",
    "ShannonEntropy": "Entropy",
}

def _load_eval_data(dataset, query_size, agent, smoothing_weight=0.0):
    if agent in ["Oracle", "UpperBound"]:
        file = os.path.join("/home/thorben/phd/projects/al-benchmark/runs", dataset, agent, "accuracies.csv")
    else:
        file = os.path.join("/home/thorben/phd/projects/al-benchmark/runs", dataset, str(query_size), agent, "accuracies.csv")
    all_runs = pd.read_csv(file, header=0, index_col=0)
    all_runs = all_runs.dropna(axis=0)
    x = list(all_runs.index)
    x = [i + get_init_pool_size(dataset) for i in x]
    mean = np.mean(all_runs.values, axis=1)
    std = np.std(all_runs.values, axis=1)
    if smoothing_weight > 0.0:
        mean, std = moving_avrg([mean, std], smoothing_weight)
    return x, mean, std

class SigmoidRegression(torch.nn.Module):
    def __init__(self, max_value):
        super(SigmoidRegression, self).__init__()
        self.max_value = max_value
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        z = self.lin(x)
        # z = z * (1 - x) # entropy function
        # z = self.max_value * torch.sigmoid(z)
        z = self.max_value / (1 + torch.exp(-(z - 2.0))) # right-shifted sigmoid
        return z


def _insert_oracle_forecast(df:pd.DataFrame):
    max_iterations = len(df)
    datasets = df["dataset"].unique()
    for dataset in datasets:
        df_oracles = df[(df["dataset"] == dataset) &
                        (df["query_size"] == 1) &    # fixed for oracles entries
                        (df["agent"] == "Oracle")
                        ]
        oracle_data = df_oracles.drop(["agent", "dataset", "query_size", "trial"], axis=1)
        lr_model = LinearRegression()
        lr_model.fit(oracle_data["iteration"].values, oracle_data["auc"].values)
        _, ub, _ = _load_eval_data(dataset, None, "UpperBound")


def _get_sigmoid_regression(x, y, x_test, upper_bound):
    max_value = max(x_test)
    x_train = torch.tensor(x).float().unsqueeze(-1)
    x_train /= max_value
    y_train = torch.from_numpy(y).float().unsqueeze(-1)
    model = SigmoidRegression(torch.tensor(upper_bound).float())
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        # if epoch % 50 == 0:
        #     print(epoch, loss.item())
    with torch.no_grad():
        x_test = torch.linspace(max(x), max(x_test), 100).reshape(-1, 1)
        # x_test_poly = poly.transform(x_test)
        return x_test, model(x_test/max_value)


def _create_plot_for_query_size(ax, dataset, query_size, y_label, title, smoothing_weight, show_auc, forecast_oracle=True):
    inferred_x_axis = None
    sorted_agents = []
    # Normal Agents
    for agent in _all_agents:
        try:
            display_name = _agent_names.get(agent, agent)
            color = _agent_colors[agent]
            x, mean, _ = _load_eval_data(dataset, query_size, agent, smoothing_weight)
            if agent != "Oracle":
                if inferred_x_axis is not None and x != inferred_x_axis:
                    print(f"[Warning]: axis of agent {agent} is different from others")
                    print(f"{agent}: {x}")
                    print(f"Previous: {inferred_x_axis}")
                    print("Ignoring...")
                else:
                    inferred_x_axis = x
            sorted_agents.append( [np.mean(mean), x, mean, color, display_name] )
        except FileNotFoundError:
            pass
    upper_bound = plot_upper_bound(ax, dataset, inferred_x_axis, "black")
    # Oracle Forecasting
    try:
        color = _agent_colors["Oracle"]
        x, mean, _ = _load_eval_data(dataset, query_size, "Oracle", smoothing_weight)
        if forecast_oracle and max(x) < max(inferred_x_axis):
            x_axis, forecast = _get_sigmoid_regression(x, mean, inferred_x_axis, upper_bound)
            ax.plot(x_axis, forecast, linewidth=1.5, c=color, alpha=0.8, linestyle="--")
    except FileNotFoundError:
        pass
    for _, x, mean, color, display_name in sorted(sorted_agents)[::-1]:
        plot_batch_benchmark(ax, x, mean, color, display_name, show_auc=show_auc)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

def full_plot(dataset, query_size=None, y_label="Accuracy", show_auc=True, smoothing_weight=0.0, radjust=0.8, forecast_oracle=True):
    if query_size is None:
        base_path = os.path.join("runs", dataset)
        query_size = list(os.listdir(base_path))
        if "Oracle" in query_size:
            query_size.remove("Oracle")
        if "UpperBound" in query_size:
            query_size.remove("UpperBound")
        query_size = sorted(query_size, key=lambda x: int(x))

    if isinstance(query_size, list):
        n_rows = math.ceil(len(query_size) / 2)
        fig, ax = plt.subplots(n_rows, 2, figsize=(12, 5*n_rows))
        ax = ax.flatten()
        legend_created = False
        for i in range(len(query_size)):
            _create_plot_for_query_size(ax[i], dataset, query_size[i], y_label if i%2==0 else "",
                                        f"{dataset} - {query_size[i]}", smoothing_weight, show_auc,
                                        forecast_oracle)
            if not legend_created:
                fig.legend(loc=7)
                legend_created = True
        fig.tight_layout()
        fig.subplots_adjust(right=radjust, hspace=0.2, wspace=0.3)
    elif isinstance(query_size, int):
        fig, ax = plt.subplots(figsize=(8, 5))
        _create_plot_for_query_size(ax, dataset, query_size, y_label, dataset, smoothing_weight, show_auc, forecast_oracle)
        plt.legend(bbox_to_anchor=(1.03, 0.5), loc="center left")
        plt.tight_layout()
    else:
        raise ValueError()


def plot_single(ax, dataset, query_size, agent, label, color,
                show_auc=True, smoothing_weight=0.0, show_std=False):
    x, mean, std = _load_eval_data(dataset, query_size, agent, smoothing_weight)
    if show_auc:
        auc = np.mean(mean)
        label += f" auc: {auc:.3f}"
    if show_std:
        label += f" +- {np.mean(std):.3f}"
        ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.5)
    ax.plot(x, mean, label=label, color=color)



def plot_batch_benchmark(ax, x, y, color, display_name, alpha=0.8, linewidth=1.5, show_auc=True):
    full_name = f"{display_name}"
    if show_auc:
        full_name += " - AUC: %1.3f"%(np.mean(y))
    ax.plot(x, y, label=full_name, linewidth=linewidth, c=color, alpha=alpha)


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


    font = {'size': 16}
    import matplotlib
    matplotlib.rc('font', **font)
    fig, axes = plt.subplots(2, 1, dpi=150, figsize=(5, 5))
    show_auc = True
    qs = 20

    _create_plot_for_query_size(axes[1], "USPS", 1, "Acc", f"USPS: Query Size {qs}",
                               smoothing_weight=0.0, show_auc=show_auc, forecast_oracle=False)
    plt.show()

    plot_single(axes, "Splice", qs, "MarginScore_scratch", label="Scratch", color="red",
                show_auc=show_auc, show_std=True)
    plot_single(axes, "Splice", qs, "MarginScore_finetuning", label="Finetuning", color="green",
                show_auc=show_auc, show_std=True)
    plot_single(axes, "Splice", qs, "MarginScore", label="Shrinking", color="blue",
                show_auc=show_auc, show_std=True)
    plot_single(axes, "Splice", qs, "MarginScore_shrinking", label="Shrinking", color="orange",
                show_auc=show_auc, show_std=True)
    axes.set_title('Splice')
    axes.set_ylabel('Accuracy')
    axes.set_xlabel('# labeled datapoints')
    plt.grid(visible=True)
    axes.legend(fontsize='x-small')
    axes.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    plt.tight_layout()
    plt.show()

    # base_path = "runs/Splice/50/TypiClust"
    # collect_results(base_path, "run_")

    # full_plot("FashionMnist", query_size=None)
    # plt.show()
