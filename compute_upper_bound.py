import yaml
import experiment_util as util
import argparse
from core.classifier import fit_and_evaluate
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--dataset", type=str, default="splice")
parser.add_argument("--encoded", type=bool, default=False)
parser.add_argument("--restarts", type=int, default=3)
args = parser.parse_args()

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    with open(f"configs/{args.dataset.lower()}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    # TODO remove again
    # class_name = "classifier_embedded" if args.encoded else "classifier"
    # config[class_name]["dropout"] = 0.2

    pool_rng = np.random.default_rng(run_id)
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(args.data_folder, config, pool_rng=pool_rng, encoded=args.encoded)
    dataset = dataset.to(util.device)

    base_path = os.path.join("runs", dataset.name, "UpperBound")
    log_path = os.path.join(base_path, f"run_{run_id}")

    save_meta_data(log_path, None, None, dataset)

    model_rng = torch.Generator()
    model_rng.manual_seed(args.model_seed + run_id)
    # some convoluted saving to make it compatible with collect_results
    # and to be consistent with logging of other runs
    accuracies = [fit_and_evaluate(dataset, model_rng)]
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
