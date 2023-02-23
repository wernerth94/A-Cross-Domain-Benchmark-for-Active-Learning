import experiment_util as util
import argparse
import numpy
from core.classifier import fit_and_evaluate
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--restarts", type=int, default=15)
args = parser.parse_args()

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    numpy.random.seed(run_id)
    torch.random.manual_seed(run_id)

    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(cache_folder=args.data_folder)
    dataset = dataset.to(util.device)

    base_path = os.path.join("runs", dataset.name, "UpperBound")
    log_path = os.path.join(base_path, f"run_{run_id}")

    save_meta_data(log_path, None, None, dataset)

    # some convoluted saving to make it compatible with collect_results
    # and to be consistent with logging of other runs
    accuracies = [fit_and_evaluate(dataset)]
    data_dict = {}
    for k, v in enumerate(accuracies):
        data_dict[k] = [v[-1]]
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(log_path, "accuracies.csv"))

    # save learning curves for training evaluation
    plot_learning_curves(accuracies, os.path.join(log_path, "learning_curves.jpg"))

    # collect results from all runs
    collect_results(base_path, "run_")
    run_id += 1