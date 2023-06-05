import os
from os.path import exists, join
import pandas as pd
import numpy as np

domains = ["Vector", "Vector Embedded", "Image", "Image Embedded", "Text"]
agents = ["RandomAgent", "Oracle", "Coreset_Greedy", "TypiClust", "MarginScore", "ShannonEntropy", "Badge", "BALD", ]
agents_no_random = agents[1:]
vector_data = ["Splice", "DNA", "USPS"]
enc_vector_data = ["SpliceEncoded", "DNAEncoded", "USPSEncoded"]
img_data = ["Cifar10", "FashionMnist"]
enc_img_data = ["Cifar10Encoded", "FashionMnistEncoded"]
text_data = ["TopV2", "News"]

def get_auc(acc_file):
    accuracies = pd.read_csv(acc_file, header=0, index_col=0)
    values = accuracies.values
    return np.sum(values, axis=0) / values.shape[0]

def generate_table(agents, datasets, out_file):
    result_table = pd.DataFrame(index=agents, columns=datasets)
    result_table["Sum"] = [0.0]*len(agents)
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
                    auc = get_auc(acc_file)
                    result_table.loc[agent, "Sum"] += np.median(auc).item()
                    result_table.loc[agent, dataset] = "%1.3f +- %1.2f"%(np.median(auc).item(), np.std(auc).item())
            else:
                print(f"{agent} is missing for {dataset}")

    if len(result_table) > 0:
        result_table = result_table.sort_values("Sum", ascending=False)
        result_table = result_table.drop(columns=["Sum"])

        result_table_file = os.path.join("results", out_file)
        if os.path.exists(result_table_file):
            os.remove(result_table_file)
        result_table.to_csv(result_table_file)


def new_list():
    return list()

def generate_domain_table(agents, out_file):
    result_table = pd.DataFrame(index=agents, columns=domains)
    result_table["All"] = [0.0]*len(agents)
    result_table["Sum"] = [0.0]*len(agents)
    all_results_per_agent = dict()
    for a in agents:
        all_results_per_agent[a] = []

    datasets_per_domain = [vector_data, enc_vector_data, img_data, enc_img_data, text_data]
    for domain, datasets in zip(domains, datasets_per_domain):
        domain_results = {}
        for a in agents:
            domain_results[a] = []
        domain_results["RandomAgent"] = [1.0]

        for dataset in datasets:
            dataset_folder = join("runs", dataset)
            acc_file = join(dataset_folder, "RandomAgent", "accuracies.csv")
            if not exists(acc_file):
                print(f"Random for {dataset} is missing")
                continue
            random_auc = np.median(get_auc(acc_file)).item()
            for agent in agents_no_random:
                acc_file = join(dataset_folder, agent, "accuracies.csv")
                if not exists(acc_file):
                    print(f"{agent} for {dataset} is missing")
                    continue
                auc = np.median(get_auc(acc_file)).item()
                domain_results[agent].append(auc / random_auc)
        for agent, perfs in domain_results.items():
            result_table.loc[agent, domain] = "%1.2f +- %1.2f"%(np.mean(perfs).item(), np.std(perfs).item())
            # result_table.loc[agent, "All"].append(perfs)
            all_results_per_agent[agent].append(perfs)
            result_table.loc[agent, "Sum"] += np.mean(perfs).item()

    if len(result_table) > 0:
        result_table = result_table.sort_values("Sum", ascending=False)
        result_table = result_table.drop(columns=["Sum"])
        for agent in agents:
            perfs = np.concatenate(all_results_per_agent[agent], axis=0)
            result_table.loc[agent, "All"] = "%1.2f +- %1.2f"%(np.mean(perfs).item(), np.std(perfs).item())

        result_table_file = os.path.join("results", out_file)
        if os.path.exists(result_table_file):
            os.remove(result_table_file)
        result_table.to_csv(result_table_file)

generate_domain_table(agents, "result_domains.csv")

# generate_table(agents, vector_data, "result_vector.csv")
# generate_table(agents, enc_vector_data, "result_enc_vector.csv")
# generate_table(agents, img_data, "result_img.csv")
# generate_table(agents, enc_img_data, "result_enc_img.csv")
# generate_table(agents, text_data, "result_text.csv")
