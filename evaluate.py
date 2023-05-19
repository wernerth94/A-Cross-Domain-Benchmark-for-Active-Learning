import time

import experiment_util as util
import argparse
from pprint import pprint
from tqdm import tqdm
import core
import yaml
import torch
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--agent", type=str, default="TypiClust")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--encoded", type=int, default=1)
# parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=10)
##########################################################
parser.add_argument("--experiment_postfix", type=str, default=None)
args = parser.parse_args()
args.encoded = bool(args.encoded)

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    with open(f"configs/{args.dataset.lower()}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    config["current_run_info"] = args.__dict__
    print("Config:")
    pprint(config)
    print("Config End \n")

    pool_rng = np.random.default_rng(args.pool_seed + run_id)
    model_seed = args.model_seed + run_id
    # This is currently the only way to seed dropout layers in Python
    torch.random.manual_seed(args.model_seed + run_id)
    data_loader_seed = 1

    AgentClass = get_agent_by_name(args.agent)
    DatasetClass = get_dataset_by_name(args.dataset)

    # Inject additional configuration into the dataset config (See BALD agent)
    AgentClass.inject_config(config)
    DatasetClass.inject_config(config)

    dataset = DatasetClass(args.data_folder, config, pool_rng, args.encoded)
    dataset = dataset.to(util.device)
    env = core.ALGame(dataset,
                      pool_rng,
                      model_seed=model_seed,
                      data_loader_seed=data_loader_seed,
                      device=util.device)
    agent = AgentClass(args.agent_seed, config)

    if args.experiment_postfix is not None:
        base_path = os.path.join("runs", dataset.name, f"{agent.name}_{args.experiment_postfix}")
    else:
        base_path = os.path.join("runs", dataset.name, agent.name)
    log_path = os.path.join(base_path, f"run_{run_id}")

    print(f"Starting run {run_id}")
    time.sleep(0.1) # prevents printing uglyness with tqdm

    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        iterator = tqdm(range(env.env.budget), miniters=2)
        for i in iterator:
            action = agent.predict(*state)
            state, reward, done, truncated, info = env.step(action.item())
            iterator.set_postfix({"accuracy": env.accuracies[1][-1]})
            if done or truncated:
                break # fail save; should not happen

    # collect results from all runs
    collect_results(base_path, "run_")
    save_meta_data(log_path, agent, env, dataset)
    run_id += 1
