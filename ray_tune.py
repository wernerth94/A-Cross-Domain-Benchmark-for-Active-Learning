import argparse
import os
from os.path import *
import yaml
import numpy as np
from core.helper_functions import get_dataset_by_name
from raytune import pretext_encoder, embedded_classification

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--dataset', type=str, default="splice")
parser.add_argument('--num_trials', type=int, default=200)
parser.add_argument('--max_conc_trials', type=int, default=15)


if __name__ == '__main__':
    args = parser.parse_args()

    benchmark_folder = "al-benchmark"
    base_path = os.path.split(os.getcwd())[0]
    cache_folder = join(base_path, "datasets")

    config_file = join(base_path, benchmark_folder, f"configs/{args.dataset}.yaml")
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)
    # check the dataset
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(cache_folder, config, np.random.default_rng(1), encoded=False)
    # output
    output_folder = join(base_path, benchmark_folder, "raytune_output")
    log_folder = join(output_folder, dataset.name)
    os.makedirs(log_folder, exist_ok=True)

    pretext_encoder.tune_pretext(args.num_trials, args.max_conc_trials, cache_folder, join(base_path, benchmark_folder), log_folder, args.dataset)
    # embedded_classification.tune_encoded_classification(args.num_trials, args.max_conc_trials, log_folder, config_file, cache_folder, DatasetClass, join(base_path, benchmark_folder))
