import itertools, math
import os
from os.path import exists, join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import t

def _find_missing_runs():
    datasets = ["Splice", "SpliceEncoded", "DNA", "DNAEncoded", "USPS", "USPSEncoded", "TopV2", "News",
                "Cifar10", "Cifar10Encoded", "FashionMnist", "FashionMnistEncoded",
                "ThreeClust", "DivergingSin"]
    agents = ["Badge", "BALD", "CoreGCN", "Coreset_Greedy", "DSA", "LSA", "MarginScore", "RandomAgent", "ShannonEntropy", "TypiClust"]
    for dataset in datasets:
        dataset_folder = join("runs", dataset)
        if not exists(dataset_folder):
            print(f"Folder missing {dataset_folder}")
            continue
        for query_size in ["1", "5", "20", "50", "100", "Oracle", "UpperBound"]:
            query_folder = join(dataset_folder, query_size)
            if not exists(query_folder):
                print(f"Folder missing {query_folder}")
                continue
            if query_size not in ["Oracle", "UpperBound"]:
                for agent in agents:
                    agent_folder = join(query_folder, agent)
                    if not exists(agent_folder):
                        print(f"Folder missing {agent_folder}")
                        continue
                    acc_file = join(agent_folder, "accuracies.csv")
                    if exists(acc_file):
                        accuracies = pd.read_csv(acc_file, header=0, index_col=0)
                        if len(accuracies.columns) < 50:
                            print(f"Missing runs for {acc_file} (found {len(accuracies.columns)})")
                    else:
                        print(f"Accuracy file missing {acc_file}")




def _t_value_for_samplesize(n_samples, sig_level= 0.95):
    return t.ppf(sig_level, n_samples)

def two_tailed_paired_t_test(df:pd.DataFrame, treatment_col, sample_col, max_sample=50):
    """
    Based on: Randomness is the Root of All Evil: More Reliable Evaluation of Deep Active Learning
    Github: https://intellisec.de/research/eval-al/
    """
    df = df[ df[sample_col] < max_sample ]
    all_agents = list(df['agent'].unique())
    t_values = []
    avg_b = df.groupby(treatment_col)
    for agent_pair in tqdm(itertools.combinations(all_agents, 2), total=int((len(all_agents)**2 - len(all_agents))/2)):
        for treatment, sub_df in avg_b:
            sub_df = sub_df.loc[(sub_df['agent'] == agent_pair[0]) | (sub_df['agent'] == agent_pair[1])]
            if len(list(sub_df['agent'].unique())) == 2:
                n_samples = len(sub_df[sample_col].unique())
                sub_df_g = sub_df.groupby(sample_col)
                acc_diff = []
                for sample, sub_sub_df in sub_df_g:
                    try:
                        acc_diff.append(sub_sub_df.loc[sub_df['agent'] == agent_pair[0]]['acc'].values[0] -
                                        sub_sub_df.loc[sub_df['agent'] == agent_pair[1]]['acc'].values[0])
                    except:
                        print(f"Problem for {agent_pair} at treatment {treatment} and sample {sample}")
                mean_difference = np.array(acc_diff).mean()
                std = np.array(acc_diff).std()
                n = len(acc_diff)
                std_error = std / math.sqrt(n)
                t_value = mean_difference / std_error
                t_value_s = t.ppf(0.95, n_samples)
                if t_value > t_value_s:
                    t_values.append([agent_pair[0], agent_pair[1], treatment, t_value, True])
                elif t_value < - t_value_s:
                    t_values.append([agent_pair[1], agent_pair[0], treatment, t_value, True])
                else:
                    t_values.append([agent_pair[0], agent_pair[1], treatment, t_value, False])
                    t_values.append([agent_pair[1], agent_pair[0], treatment, t_value, False])

    t_values_df = pd.DataFrame(t_values, columns=['M0', 'M1', treatment_col, 't_value', 'score'])
    return t_values_df


def plot_heatmap_individual(t_tables:pd.DataFrame, means, plots_path):
    """
    Based on: Randomness is the Root of All Evil: More Reliable Evaluation of Deep Active Learning
    Github: https://intellisec.de/research/eval-al/
    """
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    # fig, axn = plt.subplots(2, 1, sharey=False, figsize=(20, 8), gridspec_kw={'height_ratios': [6, 1]})
    cmap = sns.cm.rocket_r

    # plt.subplots_adjust(left=0.06, bottom=0.05, right=0.85, top=None, wspace=None, hspace=0.05)
    # bbox = axn[0].axes.get_subplotspec().get_position(fig)
    # bbox1 = axn[1].axes.get_subplotspec().get_position(fig)
    # cbar_ax = fig.add_axes([0.88, 0.05, 0.02, bbox.height + bbox1.height + 0.02])
    # cbar_ax.tick_params(labelsize=50)
    # fname = f'B4000_3trials_t.png'

    pivot_df = t_tables.pivot_table(values="score", index="M1", columns="M0", aggfunc="mean")

    h_det = sns.heatmap(data=pivot_df, robust=True, annot=True, cmap=cmap, #xticklabels=x,
                        yticklabels=False,
                        #cbar_ax=cbar_ax,
                        cbar_kws={'format': '%.2f'},
                        vmin=0, vmax=1, annot_kws={"fontsize": 10}, fmt='.2f')
    plt.tick_params(axis='both', which='major', pad=18)
    h_det.set_xticklabels(h_det.get_xticklabels(), rotation=45, fontsize=10)
    # m_h = sns.heatmap(data=means[key].transpose(), robust=True, annot=True, cmap=cmap,
    #                   xticklabels=False,
    #                   yticklabels=False, cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
    # plt.tick_params(axis='both', which='major', pad=20)

    # fpath = plots_path / fname
    # print("Saved at: ", fpath)
    # os.makedirs(plots_path, exist_ok=True)
    # plt.savefig(fpath)
    plt.tight_layout()
    plt.show()


def _query_to_list(query, current_folder):
    if query is None:
        result_list = list(os.listdir(current_folder))
    elif isinstance(query, list):
        result_list = query
    elif isinstance(query, str):
        result_list = [query]
    else:
        raise ValueError(f"Query not recognized: {query}")
    return result_list

def combine_agents_into_df(dataset=None, query_size=None, agent=None):
    base_folder = "runs"

    df_data = {
        "dataset": [],
        "query_size": [],
        "agent": [],
        "trial": [],
        "iteration": [],
        "acc": []
    }
    dataset_list = _query_to_list(dataset, base_folder)
    for dataset_name in dataset_list:
        dataset_folder = join(base_folder, dataset_name)
        query_size_list = _query_to_list(query_size, dataset_folder)
        for query_size_name in query_size_list:
            if query_size_name in ["UpperBound", "Oracle"]:
                continue
            query_size_folder = join(dataset_folder, query_size_name)
            agent_list = _query_to_list(agent, query_size_folder)
            for agent_name in agent_list:
                agent_folder = join(query_size_folder, agent_name)
                acc_file = join(agent_folder, "accuracies.csv")
                if exists(acc_file):
                    accuracies = pd.read_csv(acc_file, header=0, index_col=0).values
                    for trial in range(accuracies.shape[1]):
                        for iteration in range(accuracies.shape[0]):
                            if not np.isnan(accuracies[iteration, trial]):
                                df_data["dataset"].append(dataset_name)
                                df_data["query_size"].append(query_size_name)
                                df_data["agent"].append(agent_name)
                                df_data["trial"].append(trial)
                                df_data["iteration"].append(iteration)
                                df_data["acc"].append(accuracies[iteration, trial])
    df = pd.DataFrame(df_data)
    return df.sort_values(["dataset", "query_size", "agent", "trial", "iteration"])


def average_out_columns(df:pd.DataFrame, columns:list):
    result_df = df.copy(deep=True)
    for col in columns:
        other_columns = [c for c in result_df.columns if c not in [col, "acc"] ]
        result_list = []
        grouped_df = result_df.groupby(other_columns)
        for key, sub_df in grouped_df:
            mean = sub_df["acc"].mean()
            sub_df = sub_df.drop(col, axis=1)
            sub_df = sub_df.drop(sub_df.index[1:])
            sub_df["acc"] = mean
            result_list.append(sub_df)
        result_df = pd.concat(result_list)
    return result_df



if __name__ == '__main__':
    _find_missing_runs()
    exit(0)
    run = "runs/Splice"
    df = combine_agents_into_df(dataset="Splice")
    df = average_out_columns(df, ["iteration"])
    df = df.drop("dataset", axis=1)
    t_table = two_tailed_paired_t_test(df, treatment_col="query_size", sample_col="trial")
    heatmap_data = t_table[t_table["query_size"] == "1"]#.drop(["query_size"], axis=1)
    plot_heatmap_individual(heatmap_data, None, None)
    # plot_heatmap_individual(t_table.pivot(index="M0", columns="M1", values="t_value"), None, None)
