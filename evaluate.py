import experiment_util as util
import argparse
import numpy
from tqdm import tqdm
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--agent", type=str, default="margin")
parser.add_argument("--dataset", type=str, default="usps")
parser.add_argument("--experiment_postfix", type=str, default=None)
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=50)
args = parser.parse_args()

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    pool_rng = np.random.default_rng(args.pool_seed)# + run_id)
    agent_rng = np.random.default_rng(args.agent_seed)
    model_seed = args.model_seed + run_id
    data_loader_seed = 1
    # numpy.random.seed(run_id)
    # torch.random.manual_seed(run_id)

    AgentClass = get_agent_by_name(args.agent)
    DatasetClass = get_dataset_by_name(args.dataset)

    dataset = DatasetClass(pool_rng=pool_rng, cache_folder=args.data_folder)
    dataset = dataset.to(util.device)
    env = core.ALGame(dataset,
                      args.sample_size,
                      pool_rng,
                      model_seed=model_seed,
                      data_loader_seed=data_loader_seed,
                      device=util.device)
    agent = AgentClass(agent_rng)
    if args.experiment_postfix is not None:
        base_path = os.path.join("runs", dataset.name, f"{agent.name}_{args.experiment_postfix}")
    else:
        base_path = os.path.join("runs", dataset.name, agent.name)
    log_path = os.path.join(base_path, f"run_{run_id}")

    save_meta_data(log_path, agent, env, dataset)

    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        for i in tqdm(range(env.env.budget)):
            action = agent.predict(*state)
            state, reward, done, truncated, info = env.step(action.item())
            if done or truncated:
                break # fail save; should not happen

    # collect results from all runs
    collect_results(base_path, "run_")
    run_id += 1
