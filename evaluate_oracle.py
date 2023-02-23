import torch

import experiment_util as util
import argparse
import numpy
from tqdm import tqdm
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--dataset", type=str, default="dna")
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=1)
parser.add_argument("--store_dataset", type=bool, default=False)
args = parser.parse_args()

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    numpy.random.seed(run_id)
    torch.random.manual_seed(run_id)

    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(cache_folder=args.data_folder)
    dataset = dataset.to(util.device)
    env = core.OracleALGame(dataset,
                            args.sample_size,
                            device=util.device)
    base_path = os.path.join("runs", dataset.name, f"Oracle")
    log_path = os.path.join(base_path, f"run_{run_id}")
    save_meta_data(log_path, None, env, dataset)

    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        for i in tqdm(range(env.env.budget)):
            state, reward, done, truncated, info = env.step()
            if done or truncated:
                break # fail save; should not happen

        if args.store_dataset:
            # store dataset for later HP optimization
            out_file = os.path.join(log_path, "labeled_data.pt")
            torch.save({
                "x": env.env.x_labeled,
                "y": env.env.y_labeled
            }, out_file)


    # collect results from all runs
    collect_results(base_path, "run_")
    run_id += 1