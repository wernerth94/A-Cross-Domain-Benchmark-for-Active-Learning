import experiment_util as util
import os, argparse
from time import time
import torch
import numpy
from tqdm import tqdm
import datasets
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=50)
args = parser.parse_args()

numpy.random.seed(args.run_id)
torch.random.manual_seed(args.run_id)

DatasetClass = get_dataset_by_name(args.dataset)
dataset = DatasetClass(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = core.OracleALGame(dataset,
                        args.sample_size,
                        device=util.device)
base_path = os.path.join("runs", dataset.name, f"Oracle")
log_path = os.path.join(base_path, f"run_{args.run_id}")
save_meta_data(log_path, None, env, dataset)

with core.EnvironmentLogger(env, log_path, util.is_cluster, args.restarts) as env:
    for _ in tqdm(range(args.restarts)):
        done = False
        dataset.reset()
        state = env.reset()
        while not done:
            state, reward, done, truncated, info = env.step()

# collect results from all runs
collect_results(base_path, "run_")