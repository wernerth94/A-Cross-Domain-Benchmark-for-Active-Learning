import os
from datasets.splice import Splice
from agents.coreset import Coreset_Greedy
from core.environment import ALGame
from core.logging import EnvironmentLogger
import torch
import numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numpy.random.seed(42)
torch.random.manual_seed(42)

SAMPLE_SIZE = 20

dataset = Splice(cache_folder="../../datasets")
dataset = dataset.to(device)
env = ALGame(dataset,
             SAMPLE_SIZE,
             Coreset_Greedy.get_classifier_factory(),
             Coreset_Greedy.create_state_callback,
             device=device)
agent = Coreset_Greedy()
log_path = os.path.join("../runs", dataset.name, agent.name)

with EnvironmentLogger(env, log_path) as env:
    for _ in range(2):
        done = False
        state = env.reset()
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, info = env.step(action.item())
