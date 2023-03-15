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
print(f"Used Torch Device: {device}")
is_cluster = 'miniconda3' in sys.prefix

