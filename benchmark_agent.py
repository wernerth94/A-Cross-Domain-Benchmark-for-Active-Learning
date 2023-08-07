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

if args.experiment_postfix is not None:
    log_path = os.path.join("benchmark", f"{agent.name}_{args.experiment_postfix}")
else:
    log_path = os.path.join("benchmark", agent.name)
time.sleep(0.1) # prevents printing uglyness with tqdm

def track_ram(pid:int, out, delay=0.25):
    for i in range(10000):
        ram = psutil.Process(pid).memory_info().rss / (1024**3) # ^2=Mb ; ^3=Gb
        out[i] = ram
        # out.put(ram)
        time.sleep(delay)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
benchmark_data = pd.DataFrame(columns=["labeled points", "sample size", "ram", "time"])
sample_sizes = [50, 100, 150, 200]
for ss in sample_sizes:
    print(f"starting sample size {ss}")
    time.sleep(0.1) # prevents printing uglyness with tqdm
    x_axis = []
    ram_usage_output = multiprocessing.Array("f", range(10000))
    predict_time = []
    done = False
    dataset.reset()
    state = env.reset()
    # iterator = tqdm(range(env.budget), miniters=2)
    iterator = tqdm(range(min(500, env.budget)))
    ram_thread = multiprocessing.Process(target=track_ram, args=(os.getpid(), ram_usage_output,))
    for i in iterator:
        start_time = time.time()
        action = agent.predict(*state, sample_size=ss)
        x_axis.append(len(env.x_labeled))
        predict_time.append(time.time() - start_time)
        ram_thread.start()
        state, reward, done, truncated, info = env.step(action)
        # iterator.set_postfix({"ram": ram_usage[-1], "time": predict_time[-1]})
        if done or truncated:
            # triggered when sampling batch_size is >1
            break
    ram_thread.terminate()
    incumbent = 0.0
    ram_incumbent = []
    for i in range(10000):
        r = ram_usage_output[i]
        if r == i:
            break
        incumbent = max(incumbent, r)
        ram_incumbent.append(incumbent)
    sample_ids = np.linspace(0, len(ram_incumbent)-1, len(x_axis))
    sample_ids = sample_ids.astype(int)
    ram_incumbent = np.array(ram_incumbent)[sample_ids]

    ax1.plot(x_axis, ram_incumbent, label=f"{ss}")
    ax2.plot(x_axis, predict_time, label=f"{ss}")

    for i in range(len(x_axis)):
        benchmark_data = pd.concat([benchmark_data,
                                    pd.DataFrame({
                                        "labeled points": x_axis[i],
                                        "sample size": ss,
                                        "ram": ram_incumbent[i],
                                        "time": predict_time[i]
                                    }, index=[i])])

if os.path.exists(log_path):
    shutil.rmtree(log_path)
os.makedirs(log_path)
benchmark_data.to_csv(os.path.join(log_path, "data.csv"))

ax1.set_title("RAM (Gb)")
ax1.legend()
ax1.grid()

ax2.set_title("Time (s)")
ax2.legend()
ax2.grid()
plt.savefig(os.path.join(log_path, "plot.jpg"))
