from core.helper_functions import get_dataset_by_name
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
args = parser.parse_args()

all_names = [
    "splice",
    "dna",
    "usps",
    "fashion_mnist",
    "cifar10"
]

for name in all_names:
    print("##########################################")
    print(f"downloading {name}...")
    DatasetClass = get_dataset_by_name(name)
    dataset = DatasetClass(cache_folder=args.data_folder)

print("\n")
print("> all datasets downloaded")