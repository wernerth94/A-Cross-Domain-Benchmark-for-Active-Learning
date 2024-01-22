import experiment_util as util
import argparse
from pprint import pprint
import yaml
from tqdm import tqdm
import torch
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--dataset", type=str, default="dna")
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=3)
parser.add_argument("--store_dataset", type=bool, default=False)
parser.add_argument("--max_budget", type=int, default=2000)
parser.add_argument("--points_per_iter", type=int, default=1)
args = parser.parse_args()
args.encoded = bool(args.encoded)


run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    with open(f"configs/{args.dataset.lower()}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    if config["dataset"]["budget"] > args.max_budget:
        print(f'overwriting budget from {config["dataset"]["budget"]} to {args.max_budget}')
        config["dataset"]["budget"] = args.max_budget
    config["current_run_info"] = args.__dict__
    print("Config:")
    pprint(config)
    print("Config End \n")

    print(f"Starting run {run_id}")
    pool_rng = np.random.default_rng(args.pool_seed + run_id)
    model_seed = args.model_seed + run_id
    # This is currently the only way to seed dropout layers in Python
    torch.random.manual_seed(args.model_seed + run_id)
    data_loader_seed = 1

    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(args.data_folder, config, pool_rng, args.encoded)
    dataset = dataset.to(util.device)
    env = core.OracleALGame(dataset,
                            args.sample_size,
                            pool_rng,
                            model_seed=model_seed,
                            data_loader_seed=data_loader_seed,
                            points_added_per_round=args.points_per_iter,
                            device=util.device)
    base_path = os.path.join("runs", dataset.name, f"Oracle")
    log_path = os.path.join(base_path, f"run_{run_id}")

    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        for i in tqdm(range(env.env.budget)):
            state, reward, done, truncated, info = env.step()
            if done or truncated:
                break # fail save; should not happen

    save_meta_data(log_path, None, env, dataset, config)
    if args.store_dataset:
        # store dataset for later HP optimization
        out_file = os.path.join(log_path, "labeled_data.pt")
        torch.save({
            "x_train": env.env.x_labeled, # specific naming convention to
            "y_train": env.env.y_labeled  # be consistent with normal dataset files
        }, out_file)


    # collect results from all runs
    collect_results(base_path, "run_")
    run_id += 1
