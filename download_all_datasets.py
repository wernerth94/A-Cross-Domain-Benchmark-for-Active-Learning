from core.helper_functions import get_dataset_by_name
import argparse
import yaml
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--encode", type=int, default=0)
args = parser.parse_args()
args.encode = bool(args.encode)

all_names = [
    "splice",
    "dna",
    "usps",
    "fashionmnist",
    "cifar10",
    "TopV2",
    "News"
]

for name in all_names:
    print("##########################################")
    print(f"downloading {name}...")
    name = name.lower()
    with open(f"configs/{name}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    pool_rng = np.random.default_rng(1)
    DatasetClass = get_dataset_by_name(name)
    DatasetClass(args.data_folder, config, pool_rng, encoded=False)
    if args.encode:
        try:
            DatasetClass(args.data_folder, config, pool_rng, encoded=True)
        except AssertionError:
            pass

print("\n")
print("> all datasets prepared")
