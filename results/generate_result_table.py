import os
from os.path import exists, join
import pandas as pd
import numpy as np

def generate_table(agents, datasets, out_file):
    result_table = pd.DataFrame(index=agents, columns=datasets)
    for dataset in datasets:
        dataset_folder = os.path.join("runs", dataset)
        if not os.path.isdir(dataset_folder):
            continue
        agent_aucs = dict()
        for a in agents:
            agent_aucs[a] = "-"

        for agent in agents:
            agent_folder = join(dataset_folder, agent)
            if exists(agent_folder):
                acc_file = join(agent_folder, "accuracies.csv")
                if exists(acc_file):
                    accuracies = pd.read_csv(acc_file, header=0, index_col=0)
                    values = accuracies.values
                    auc = np.sum(values, axis=0) / values.shape[0]
                    result_table[dataset][agent] = "%1.3f +- %1.2f"%(np.mean(auc).item(), np.std(auc).item())
            else:
                print(f"{agent} is missing for {dataset}")

    if len(result_table) > 0:
        result_table_file = os.path.join("runs", out_file)
        if os.path.exists(result_table_file):
            os.remove(result_table_file)
        result_table.to_csv(result_table_file)


agents = ["Oracle", "SAL_23_01_16", "Coreset_Greedy", "TypiClust", "MarginScore", "ShannonEntropy", "RandomAgent", "Badge", "BALD", ]
vector_data = ["Splice", "DNA", "USPS"]
img_data = ["Cifar10Encoded", "FashionMnistEncoded"]
text_data = ["TopV2", "News"]

generate_table(agents, vector_data, "result_vector.csv")
generate_table(agents, img_data, "result_img.csv")
generate_table(agents, text_data, "result_text.csv")
