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
import numpy as np
from core.helper_functions import *

def getCurrentMemoryUsage():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--agent", type=str, default="coregcn")
##########################################################
parser.add_argument("--experiment_postfix", type=str, default=None)
args = parser.parse_args()
dataset = "cifar10"

with open(f"configs/{dataset}.yaml", 'r') as f:
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
DatasetClass = get_dataset_by_name(dataset)

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

stop_flag = False
x_axis = []
ram_usage = multiprocessing.Array("f", range(10000))
predict_time = []
def track_ram(pid:int, out, delay=0.5):
    for i in range(10000):
        ram = psutil.Process(pid).memory_info().rss / (1024**3) # ^2=Mb ; ^3=Gb
        out[i] = ram
        # out.put(ram)
        time.sleep(delay)

done = False
dataset.reset()
state = env.reset()
# iterator = tqdm(range(env.budget), miniters=2)
iterator = tqdm(range(min(500, env.budget)))
ram_thread = multiprocessing.Process(target=track_ram, args=(os.getpid(), ram_usage,))
ram_thread.start()
for i in iterator:
    time.sleep(2)
    start_time = time.time()
    action = agent.predict(*state)
    x_axis.append(len(env.x_labeled))
    predict_time.append(time.time() - start_time)
    state, reward, done, truncated, info = env.step(action)
    # iterator.set_postfix({"ram": ram_usage[-1], "time": predict_time[-1]})
    if done or truncated:
        # triggered when sampling batch_size is >1
        break
ram_thread.terminate()
l_ram_usage = []
for i in range(10000):
    r = ram_usage[i]
    if r == i:
        break
    l_ram_usage.append(r)

if os.path.exists(log_path):
    shutil.rmtree(log_path)
os.makedirs(log_path)
with open(os.path.join(log_path, "statistics.txt"), "w") as f:
    f.write(f"maximal RAM (G) usage:\n")
    f.write(f"{max(ram_usage)}:\n\n")
    window = 100
    f.write(f"final predict time (mean of last {window}):\n")
    f.write(f"{sum(predict_time[-window:])/window}:\n\n")
np.savez(os.path.join(log_path, "data.npz"), x=x_axis, ram=ram_usage, time=predict_time)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
ax1.plot(l_ram_usage, label="RAM (G)", color='tab:blue')
ax1.legend()
ax1.grid()

ax2.plot(x_axis, predict_time, label="Time (s)", color='tab:red')
ax2.legend()
ax2.grid()
plt.savefig(os.path.join(log_path, "plot.jpg"))
