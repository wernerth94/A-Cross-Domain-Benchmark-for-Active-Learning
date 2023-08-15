import sys

import experiment_util as util
import time, os, shutil
import multiprocessing
import psutil
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import core
import yaml
import pandas as pd
import numpy as np
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--agent", type=str, default="margin")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--ram_interval", type=float, default=0.25)
##########################################################
parser.add_argument("--experiment_postfix", type=str, default=None)
args = parser.parse_args()

with open(f"configs/{args.dataset.lower()}.yaml", 'r') as f:
    config = yaml.load(f, yaml.Loader)
config["current_run_info"] = args.__dict__
print("Config:")
pprint(config)
print("Config End \n")

pool_rng = np.random.default_rng(args.pool_seed)
model_seed = args.model_seed
# This is currently the only way to seed dropout masks in Python
torch.random.manual_seed(args.model_seed)
# Seed numpy-based algorithms like KMeans
np.random.seed(args.model_seed)
data_loader_seed = 1

AgentClass = get_agent_by_name(args.agent)
DatasetClass = get_dataset_by_name(args.dataset)

# Inject additional configuration into the dataset config (See BALD agent)
AgentClass.inject_config(config)
DatasetClass.inject_config(config)

dataset = DatasetClass(args.data_folder, config, pool_rng, encoded=False)
dataset = dataset.to(util.device)
env = core.ALGame(dataset,
                  pool_rng,
                  model_seed=model_seed,
                  data_loader_seed=data_loader_seed,
                  device=util.device)
agent = AgentClass(args.agent_seed, config)

ss = 10000.
jump = 5000.
for trials in range(6):
    try:
        print(f"sample size {ss} ... ", end="")
        x_axis = []
        done = False
        dataset.reset()
        state = env.reset()
        for i in range(env.budget):
            action = agent.predict(*state, sample_size=int(ss))
            state, reward, done, truncated, info = env.step(action)
            if done or truncated:
                # triggered when sampling batch_size is >1
                break
        print("success")
        if ss == 10000:
            print("early stopping")
            sys.exit(0)
        else:
            ss += jump
            jump /= 2
    except Exception as ex:
        print("failed")
        ss -= jump
        jump /= 2
