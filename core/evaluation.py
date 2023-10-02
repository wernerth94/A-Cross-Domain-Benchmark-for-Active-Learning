import itertools, math
import os
from os.path import exists, join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

def two_tailed_paired_t_test(df:pd.DataFrame):
    """
    Based on: Randomness is the Root of All Evil: More Reliable Evaluation of Deep Active Learning
    Github: https://intellisec.de/research/eval-al/
    """
    all_agents = list(df['agent'].unique())
    all_iterations = list(df['iteration'].unique())
    max_iter = max(all_iterations)
    t_values = []
    n_N = 0
    n_N_scores_dict = {}
    for m_pair in tqdm(itertools.combinations(all_agents, 2)):
        avg_b = df.groupby(['iteration'])
        for (N_label), sub_df in avg_b:
            sub_df = sub_df.loc[(sub_df['agent'] == m_pair[0]) | (sub_df['agent'] == m_pair[1])]
            if len(list(sub_df['agent'].unique())) == 2:
                n_N += 1
                sub_df_g = sub_df.groupby(['trial'])
                acc_diff = []
                for _, sub_sub_df in sub_df_g:
                    acc_diff.append(sub_sub_df.loc[sub_df['agent'] == m_pair[0]]['acc'].values[0] -
                                    sub_sub_df.loc[sub_df['agent'] == m_pair[1]]['acc'].values[0])
                mean_difference = np.array(acc_diff).mean()
                std = np.array(acc_diff).std()
                n = len(acc_diff)
                std_error = std / math.sqrt(n)
                t_value = mean_difference / std_error
                # t_value_s = 2.920  # trials 3
                t_value_s = 1.676  # trials 50
                if t_value > t_value_s:
                    t_values.append([m_pair[0], m_pair[1], N_label, t_value, True])
                elif t_value < - t_value_s:
                    t_values.append([m_pair[1], m_pair[0], N_label, t_value, True])
                else:
                    t_values.append([m_pair[0], m_pair[1], N_label, t_value, False])
                    t_values.append([m_pair[1], m_pair[0], N_label, t_value, False])
        # n_N_scores_dict[BS] = n_N
        n_N = 0

    t_values_df = pd.DataFrame(t_values, columns=['M0', 'M1', 'iteration', 't_value', 'score'])
    return t_values_df, n_N_scores_dict


def plot_heatmap_individual(t_tables, means, plots_path):
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

    h_det = sns.heatmap(data=t_tables, robust=True, annot=True, cmap=cmap, #xticklabels=x,
                        yticklabels=False,
                        #cbar_ax=cbar_ax,
                        cbar_kws={'format': '%.2f'},
                        vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
    plt.tick_params(axis='both', which='major', pad=18)
    h_det.set_xticklabels(h_det.get_xticklabels(), rotation=0, fontsize=50)
    # m_h = sns.heatmap(data=means[key].transpose(), robust=True, annot=True, cmap=cmap,
    #                   xticklabels=False,
    #                   yticklabels=False, cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
    # plt.tick_params(axis='both', which='major', pad=20)

    # fpath = plots_path / fname
    # print("Saved at: ", fpath)
    # os.makedirs(plots_path, exist_ok=True)
    # plt.savefig(fpath)
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
        "agent": [],
        "trial": [],
        "acc": []
    }
    dataset_list = _query_to_list(dataset, base_folder)
    for dataset in dataset_list:
        dataset_folder = join(base_folder, dataset)
        query_size_list = _query_to_list(query_size, dataset_folder)
        for query_size in query_size_list:
            if query_size in ["UpperBound"]:
                continue
            query_size_folder = join(dataset_folder, query_size)
            agent_list = _query_to_list(agent, query_size_folder)
            for agent in agent_list:
                agent_folder = join(query_size_folder, agent)
                acc_file = join(agent_folder, "accuracies.csv")
                if exists(acc_file):
                    accuracies = pd.read_csv(acc_file, header=0, index_col=0).values
                    # accuracies = np.mean(accuracies, axis=1)
                    for trial in range(accuracies.shape[1]):
                        #for iteration in range(accuracies.shape[0]):
                        df_data["dataset"].append(dataset)
                        df_data["agent"].append(agent)
                        df_data["trial"].append(trial)
                        df_data["acc"].append(accuracies[:, trial])
    df = pd.DataFrame(df_data)
    return df



if __name__ == '__main__':
    run = "runs/Splice"
    df = combine_agents_into_df(dataset="Splice")
    t_table, _ = two_tailed_paired_t_test(df)
    plot_heatmap_individual(t_table, None, None)
