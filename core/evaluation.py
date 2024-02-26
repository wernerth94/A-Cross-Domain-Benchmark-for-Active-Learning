import itertools, math
import os
from os.path import exists, join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import t
from helper_functions import _insert_oracle_forecast

name_corrections = {
    "RandomAgent": "Random",
    "Coreset_Greedy": "Coreset",
    "ShannonEntropy": "Entropy",
    "MarginScore": "Margin"
}

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
        for query_size in ["1", "5", "20", "50", "100", "500", "Oracle", "UpperBound"]:
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
                    # Check individual runs
                    for i in range(1, 51):
                        run_folder = join(agent_folder, f"run_{i}")
                        if not exists(run_folder):
                            print(f"Folder missing {run_folder}")
                            continue
                        run_acc_file = join(run_folder, "accuracies.csv")
                        if not exists(run_acc_file):
                            print(f"Accuracy file missing {run_acc_file}")
                    # Check collection of runs
                    acc_file = join(agent_folder, "accuracies.csv")
                    if exists(acc_file):
                        accuracies = pd.read_csv(acc_file, header=0, index_col=0)
                        if len(accuracies.columns) < 50:
                            print(f"Missing runs for {acc_file} (found {len(accuracies.columns)})")
                        target_count = None
                        for c in accuracies.columns:
                            count_non_zero = (accuracies[c] != 0).sum()
                            if target_count is None:
                                target_count = count_non_zero
                            if count_non_zero != target_count:
                                print(f"Uneven number of values in run {c} of {agent_folder}")
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

def generate_full_overview(precision=2):
    datasets_raw = ["Splice", "DNA", "USPS", "Cifar10", "FashionMnist", "TopV2", "News",]
                    #"DivergingSin", "ThreeClust"]
    datasets_encoded = ["SpliceEncoded", "DNAEncoded", "USPSEncoded",
                        "Cifar10Encoded", "FashionMnistEncoded"]
    df_raw = combine_agents_into_df(dataset=datasets_raw, include_oracle=True)
    df_raw = average_out_columns(df_raw, ["iteration", "query_size"])
    df_raw = compute_ranks_over_trials(df_raw)

    df_enc = combine_agents_into_df(dataset=datasets_encoded, include_oracle=True)
    df_enc = average_out_columns(df_enc, ["iteration", "query_size"])
    df_enc = compute_ranks_over_trials(df_enc)

    leaderboard = average_out_columns(df_raw, ["dataset"]).sort_values("rank")
    intersection = leaderboard["agent"].isin(df_enc["agent"])
    leaderboard = leaderboard[intersection]
    leaderboard.index = leaderboard["agent"]
    leaderboard = leaderboard.drop(["agent", "auc"], axis=1)
    # add single unencoded datasets
    for dataset in datasets_raw:
        values = []
        for index, _ in leaderboard.iterrows():
            r = df_raw[(df_raw["agent"] == index) & (df_raw["dataset"] == dataset)]["rank"]
            if len(r) == 0:
                print(f"No runs found for {index} on {dataset}")
                continue
            values.append(round(r.item(), precision))
        leaderboard[dataset] = values
    leaderboard["Unencoded"] = leaderboard["rank"].round(precision)
    leaderboard = leaderboard.drop(["rank"], axis=1)
    # add average of all encoded datasets
    df_enc = average_out_columns(df_enc, ["dataset"])
    values = []
    for index, _ in leaderboard.iterrows():
        r = df_enc[(df_enc["agent"] == index)]["rank"]
        values.append(round(r.item(), precision))
    leaderboard["Encoded"] = values
    return leaderboard


def compute_ranks_over_trials(df:pd.DataFrame):
    assert "trial" in df.columns
    df["rank"] = df.groupby(["dataset", "trial"])["auc"].rank(ascending=False)
    df = average_out_columns(df, ["trial"])
    return df



def combine_agents_into_df(dataset=None, query_size=None, agent=None,
                           max_loaded_runs=None, include_oracle=False):
    def _load_trials_for_agent(dataset_name, query_size_name, agent_name):
        if query_size_name is not None:
            agent_folder = join(base_folder, dataset_name, query_size_name, agent_name)
        else:
            agent_folder = join(base_folder, dataset_name, agent_name)
        acc_file = join(agent_folder, "accuracies.csv")
        if exists(acc_file):
            accuracies = pd.read_csv(acc_file, header=0, index_col=0).values
            if max_loaded_runs is not None:
                N = max_loaded_runs
            else:
                N = accuracies.shape[1]
            for trial in range(N):
                for iteration in range(accuracies.shape[0]):
                    if not np.isnan(accuracies[iteration, trial]):
                        df_data["dataset"].append(dataset_name)
                        if query_size_name is not None:
                            df_data["query_size"].append(query_size_name)
                        else:
                            df_data["query_size"].append(1)
                        df_data["agent"].append(name_corrections.get(agent_name, agent_name))
                        df_data["trial"].append(trial)
                        df_data["iteration"].append(iteration)
                        df_data["auc"].append(accuracies[iteration, trial])


    base_folder = "runs"

    df_data = {
        "dataset": [],
        "query_size": [],
        "agent": [],
        "trial": [],
        "iteration": [],
        "auc": []
    }
    dataset_list = _query_to_list(dataset, base_folder)
    for dataset_name in tqdm(dataset_list):
        dataset_folder = join(base_folder, dataset_name)
        query_size_list = _query_to_list(query_size, dataset_folder)
        for query_size_name in query_size_list:
            if query_size_name in ["UpperBound", "Oracle"]:
                continue
            query_size_folder = join(dataset_folder, query_size_name)
            agent_list = _query_to_list(agent, query_size_folder)
            for agent_name in agent_list:
                _load_trials_for_agent(dataset_name, query_size_name, agent_name)
        if include_oracle:
            _load_trials_for_agent(dataset_name, None, "Oracle")
    df = pd.DataFrame(df_data)
    df = df.sort_values(["dataset", "query_size", "agent", "trial", "iteration"])
    if include_oracle:
        df = _insert_oracle_forecast(df)
    return df


def average_out_columns(df:pd.DataFrame, columns:list):
    result_df = df.copy(deep=True)
    for col in columns:
        other_columns = [c for c in result_df.columns if c not in [col, "auc", "rank"] ]
        result_list = []
        grouped_df = result_df.groupby(other_columns)
        for key, sub_df in grouped_df:
            mean = sub_df["auc"].mean()
            sub_df["auc"] = mean
            if "rank" in df.columns:
                mean = sub_df["rank"].mean()
                sub_df["rank"] = mean
            sub_df = sub_df.drop(col, axis=1) # drop averaged col from sub-dataframe
            sub_df = sub_df.drop(sub_df.index[1:]) # drop all the other useless rows
            result_list.append(sub_df)
        result_df = pd.concat(result_list)
    return result_df



if __name__ == '__main__':
    # combine_agents_into_df(["Cifar10", "FashionMnist"], include_oracle=True)

    leaderboard = generate_full_overview()
    leaderboard.to_csv("results/overview.csv")

    # _find_missing_runs()

    # run = "runs/Splice"
    # df = combine_agents_into_df(dataset="Splice")
    # df = average_out_columns(df, ["iteration"])
    # df = df.drop("dataset", axis=1)
    # t_table = two_tailed_paired_t_test(df, treatment_col="query_size", sample_col="trial")
    # heatmap_data = t_table[t_table["query_size"] == "1"]#.drop(["query_size"], axis=1)
    # plot_heatmap_individual(heatmap_data, None, None)
    # plot_heatmap_individual(t_table.pivot(index="M0", columns="M1", values="t_value"), None, None)


