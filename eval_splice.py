import experiment_util as util
import os, argparse
from agents.random_agent import RandomAgent
from agents.shannon_entropy import ShannonEntropy
from agents.margin import MarginScore
from agents.coreset import Coreset_Greedy
from agents.sar import SAR
from core.environment import ALGame
from core.logging import EnvironmentLogger
import torch
import numpy
from tqdm import tqdm
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, default="agent")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.agent == "random":
    AgentClass = RandomAgent
elif args.agent == "entropy":
    AgentClass = ShannonEntropy
elif args.agent == "margin":
    AgentClass = MarginScore
elif args.agent == "coreset":
    AgentClass = Coreset_Greedy
elif args.agent == "agent":
    AgentClass = SAR
else:
    raise ValueError(f"Agent name '{args.agent}' not recognized")

numpy.random.seed(args.seed)
torch.random.manual_seed(args.seed)

SAMPLE_SIZE = 20

# dataset = datasets.dna.DNA(cache_folder="../datasets")
dataset = datasets.Splice(cache_folder="../datasets")
# dataset = datasets.Cifar10(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = ALGame(dataset,
             SAMPLE_SIZE,
             AgentClass.create_state_callback,
             device=util.device)
agent = AgentClass()
log_path = os.path.join("runs", dataset.name, agent.name)

util.save_meta_data(log_path, agent, env, dataset)

with EnvironmentLogger(env, log_path, util.is_cluster) as env:
    for _ in tqdm(range(50)):
        done = False
        state = env.reset()
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, info = env.step(action.item())
