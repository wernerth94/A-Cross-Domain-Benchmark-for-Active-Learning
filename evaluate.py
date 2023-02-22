import experiment_util as util
import argparse
import numpy
from tqdm import tqdm
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--agent", type=str, default="entropy")
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=50)
args = parser.parse_args()

for run_id in tqdm(range(args.run_id, args.restarts)):
    numpy.random.seed(run_id)
    torch.random.manual_seed(run_id)

    AgentClass = get_agent_by_name(args.agent)
    DatasetClass = get_dataset_by_name(args.dataset)

    dataset = DatasetClass(cache_folder="../datasets")
    dataset = dataset.to(util.device)
    env = core.ALGame(dataset,
                      args.sample_size,
                      device=util.device)
    agent = AgentClass()
    base_path = os.path.join("runs", dataset.name, agent.name)
    log_path = os.path.join(base_path, f"run_{run_id}")

    save_meta_data(log_path, agent, env, dataset)

    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        while not done:
            action = agent.predict(*state)
            state, reward, done, truncated, info = env.step(action.item())

    # collect results from all runs
    collect_results(base_path, "run_")