import experiment_util as util
import argparse
import numpy
from core.classifier import fit_and_evaluate
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--sub_run_id", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--restarts", type=int, default=10)
args = parser.parse_args()

DatasetClass = get_dataset_by_name(args.dataset)

if args.sub_run_id is not None:
    print(f"Sub-run ID {args.sub_run_id} given. This will override the seed")
    numpy.random.seed(args.sub_run_id)
    torch.random.manual_seed(args.sub_run_id)
else:
    numpy.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

dataset = DatasetClass(cache_folder="../datasets")
dataset = dataset.to(util.device)

base_path = os.path.join("runs", dataset.name, "UpperBound")
log_path = base_path
if args.sub_run_id is not None:
    log_path = os.path.join(log_path, f"run_{args.sub_run_id}")

save_meta_data(log_path, None, None, dataset)

accuracies = []
for _ in range(args.restarts):
    acc = fit_and_evaluate(dataset)
    accuracies.append(acc)
data_dict = {}
for k, v in enumerate(accuracies):
    data_dict[k] = [v]
df = pd.DataFrame(data_dict)
df.to_csv(os.path.join(log_path, "accuracies.csv"))

# collect results from all runs
if args.sub_run_id is not None:
    collect_results(base_path, "run_")