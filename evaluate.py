import experiment_util as util
import argparse
import numpy
from tqdm import tqdm
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--sub_run_id", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--agent", type=str, default="margin")
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=50)
args = parser.parse_args()

AgentClass = get_agent_by_name(args.agent)
DatasetClass = get_dataset_by_name(args.dataset)

if args.sub_run_id is not None:
    print("Sub-run ID given. This will override the seed")
    numpy.random.seed(args.sub_run_id)
    torch.random.manual_seed(args.sub_run_id)
else:
    numpy.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

dataset = DatasetClass(cache_folder="../datasets")
dataset = dataset.to(util.device)
env = core.ALGame(dataset,
                  args.sample_size,
                  AgentClass.create_state_callback,
                  device=util.device)
agent = AgentClass()
base_path = os.path.join("runs", dataset.name, agent.name)
log_path = base_path
if args.sub_run_id is not None:
    log_path = os.path.join(log_path, f"run_{args.sub_run_id}")

save_meta_data(log_path, agent, env, dataset)

with core.EnvironmentLogger(env, log_path, util.is_cluster, args.restarts) as env:
    for _ in tqdm(range(args.restarts)):
        done = False
        state = env.reset()
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, info = env.step(action.item())

# collect results from all runs
if args.sub_run_id is not None:
    collect_results(base_path, "run_")