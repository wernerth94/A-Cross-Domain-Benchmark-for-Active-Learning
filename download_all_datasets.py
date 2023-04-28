from core.helper_functions import get_dataset_by_name
import argparse
import yaml
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
args = parser.parse_args()

all_names = [
    "splice",
    "dna",
    "usps",
    "fashion_mnist",
    "cifar10",
    "TopV2"
]

for name in all_names:
    print("##########################################")
    print(f"downloading {name}...")
    with open(f"configs/{name}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    pool_rng = np.random.default_rng(1)
    DatasetClass = get_dataset_by_name(name)
    dataset = DatasetClass(args.data_folder, config, pool_rng, encoded=False)

print("\n")
print("> all datasets downloaded")
