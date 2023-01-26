import experiment_util as util
import os, argparse
import torch
import numpy
from tqdm import tqdm
import agents
import datasets
import core
from core.helper_functions import save_meta_data

parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, default="margin")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.agent == "random":
    AgentClass = agents.RandomAgent
elif args.agent == "entropy":
    AgentClass = agents.ShannonEntropy
elif args.agent == "margin":
    AgentClass = agents.MarginScore
elif args.agent == "coreset":
    AgentClass = agents.Coreset_Greedy
elif args.agent == "agent":
    AgentClass = agents.SAR
else:
    raise ValueError(f"Agent name '{args.agent}' not recognized")

numpy.random.seed(args.seed)
torch.random.manual_seed(args.seed)

SAMPLE_SIZE = 20
CROSS_VALIDATION = 50

# dataset = datasets.dna.DNA(cache_folder="../datasets")
dataset = datasets.Splice(cache_folder="../datasets")
# dataset = datasets.Cifar10(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = core.ALGame(dataset,
                  SAMPLE_SIZE,
                  AgentClass.create_state_callback,
                  device=util.device)
agent = AgentClass()
log_path = os.path.join("runs", dataset.name, agent.name)

save_meta_data(log_path, agent, env, dataset)

with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
    for _ in tqdm(range(CROSS_VALIDATION)):
        done = False
        state = env.reset()
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, info = env.step(action.item())
