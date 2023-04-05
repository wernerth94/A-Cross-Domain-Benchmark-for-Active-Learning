import experiment_util as util
import argparse
import yaml
from tqdm import tqdm
import core
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--encoded", type=bool, default=True)
parser.add_argument("--sample_size", type=int, default=20)
parser.add_argument("--restarts", type=int, default=3)
parser.add_argument("--store_dataset", type=bool, default=False)
args = parser.parse_args()

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    with open(f"configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    pool_rng = np.random.default_rng(args.pool_seed + run_id)
    model_seed = args.model_seed + run_id
    data_loader_seed = 1

    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(args.data_folder, config, pool_rng, args.encoded)
    dataset = dataset.to(util.device)
    env = core.OracleALGame(dataset,
                            args.sample_size,
                            pool_rng,
                            model_seed=model_seed,
                            data_loader_seed=data_loader_seed,
                            device=util.device)
    base_path = os.path.join("runs", dataset.name, f"Oracle")
    log_path = os.path.join(base_path, f"run_{run_id}")
    save_meta_data(log_path, None, env, dataset)

    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        for i in tqdm(range(env.env.budget)):
            state, reward, done, truncated, info = env.step()
            if done or truncated:
                break # fail save; should not happen

        if args.store_dataset:
            # store dataset for later HP optimization
            out_file = os.path.join(log_path, "labeled_data.pt")
            torch.save({
                "x_train": env.env.x_labeled, # specific naming convention to
                "y_train": env.env.y_labeled  # be consistent with normal dataset files
            }, out_file)


    # collect results from all runs
    collect_results(base_path, "run_")
    run_id += 1
