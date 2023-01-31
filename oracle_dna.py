import experiment_util as util
import os, argparse
from time import time
import torch
import numpy
from tqdm import tqdm
import datasets
import core
from core.helper_functions import save_meta_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

numpy.random.seed(args.seed)
torch.random.manual_seed(args.seed)

SAMPLE_SIZE = 20
CROSS_VALIDATION = 1

dataset = datasets.dna.DNA(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = core.OracleALGame(dataset,
                        SAMPLE_SIZE,
                        device=util.device)
log_path = os.path.join("runs", dataset.name, f"Oracle_{time()}")

save_meta_data(log_path, None, env, dataset)

with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
    for _ in tqdm(range(CROSS_VALIDATION)):
        done = False
        state = env.reset()
        while not done:
            state, reward, done, truncated, info = env.step()
