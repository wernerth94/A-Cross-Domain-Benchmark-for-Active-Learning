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
parser.add_argument("--sub_run_id", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=50)
args = parser.parse_args()

if args.sub_run_id is not None:
    print("Sub-run ID given. This will override the seed")
    numpy.random.seed(args.sub_run_id)
    torch.random.manual_seed(args.sub_run_id)
else:
    numpy.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

DatasetClass = get_dataset_by_name(args.dataset)
dataset = DatasetClass(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = core.OracleALGame(dataset,
                        args.sample_size,
                        device=util.device)
base_path = os.path.join("runs", dataset.name, f"Oracle")
log_path = base_path
if args.sub_run_id is not None:
    log_path = os.path.join(log_path, f"run_{args.sub_run_id}")
save_meta_data(log_path, None, env, dataset)

with core.EnvironmentLogger(env, log_path, util.is_cluster, args.restarts) as env:
    for _ in tqdm(range(args.restarts)):
        done = False
        state = env.reset()
        while not done:
            state, reward, done, truncated, info = env.step()

# collect results from all runs
if args.sub_run_id is not None:
    collect_results(base_path, "run_")