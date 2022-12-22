import experiment_util as util
import os, argparse
from datasets.splice import Splice
from agents.random_agent import RandomAgent
from agents.shannon_entropy import ShannonEntropy
from agents.margin import MarginScore
from agents.coreset import Coreset_Greedy
from core.environment import ALGame
from core.logging import EnvironmentLogger
import torch
import numpy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, default="margin")
args = parser.parse_args()

if args.agent == "random":
    AgentClass = RandomAgent
elif args.agent == "entropy":
    AgentClass = ShannonEntropy
elif args.agent == "margin":
    AgentClass = MarginScore
elif args.agent == "coreset":
    AgentClass = Coreset_Greedy
else:
    raise ValueError(f"Agent name '{args.agent}' not recognized")

numpy.random.seed(42)
torch.random.manual_seed(42)

SAMPLE_SIZE = 20

dataset = Splice(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = ALGame(dataset,
             SAMPLE_SIZE,
             AgentClass.create_state_callback,
             device=util.device)
agent = AgentClass()
log_path = os.path.join("runs", dataset.name, agent.name)

with EnvironmentLogger(env, log_path) as env:
    for _ in tqdm(range(50)):
        done = False
        state = env.reset()
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, info = env.step(action.item())
