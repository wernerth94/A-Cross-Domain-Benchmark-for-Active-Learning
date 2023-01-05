import sys
# path additions for the cluster
sys.path.append("agents")
sys.path.append("core")
sys.path.append("datasets")
print(F"updated path is {sys.path}")

import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}\n")

# fixing hdf5 file writing on the cluster
import os
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_cluster = 'miniconda3' in sys.prefix

def save_meta_data(logpath, agent, env, dataset):
    if not os.path.exists(logpath):
        os.makedirs(logpath, exist_ok=True)
    file = os.path.join(logpath, "meta.txt")
    if os.path.exists(file):
        os.remove(file)

    with open(file, "w") as f:
        if hasattr(dataset, "get_meta_data"):
            f.write("# Dataset: \n")
            f.write(f"{dataset.get_meta_data()} \n\n")
        if hasattr(agent, "get_meta_data"):
            f.write("# Agent: \n")
            f.write(f"{agent.get_meta_data()} \n\n")
        if hasattr(env, "get_meta_data"):
            f.write("# Environment: \n")
            f.write(f"{env.get_meta_data()} \n\n")