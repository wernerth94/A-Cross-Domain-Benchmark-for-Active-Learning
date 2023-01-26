import experiment_util as util
import pandas as pd
import argparse, os, time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from core.classifier import fit_and_evaluate
from core.data import BaseDataset

import botorch
# botorch.settings.debug._set_state(True)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trials", type=int, default=200)
parser.add_argument("-r", "--runs_per_trial", type=int, default=5)

bounds = torch.Tensor([
    [1e-6, 5e-2],  # LR
    [0.0, 1e-6],  # Weight Decay
    [12, 256],  # Batch Size
    [3, 6],  # Number of Hidden Layers
    [-100, 100],
    # Hidden size multiplier: This sets the size of each hidden layer by
    # size = layer number x multiplier      or
    # size = (1/layer number) x -multiplier  if multiplier < 0
]).T

norm_bounds = torch.stack([
    torch.zeros(bounds.shape[1]),
    torch.ones(bounds.shape[1]),
])
def normalize(trial):
    # Normalizes the HP values to be [0..1]
    with torch.no_grad():
        min_ = bounds[0, :]
        max_ = bounds[1, :]
        return (trial - min_) / (max_ - min_)

def de_normalize(trial):
    # De-normalizes the HP values to report the actual values
    with torch.no_grad():
        min_ = bounds[0, :]
        max_ = bounds[1, :]
        return trial * (max_ - min_) + min_

def to_dict(trial)->dict:
    """
    Applies cleaning to the selected HP values like setting the number hidden layers to int, etc.
    :param trial:
    :return: cleaned values for trial
    """
    with torch.no_grad():
        trial = de_normalize(trial)
        hp_dict = {}
        hp_dict["lr"] = trial[0].item()
        hp_dict["weight_decay"] = trial[1].item()
        hp_dict["batch_size"]  = int(torch.round(trial[2]).item())
        num_hidden = int(torch.round(trial[3]).item()) # num hidden layers
        # hidden layer sizes
        mult = float(trial[4].item())
        if mult >= 0:
            hp_dict["hidden_sizes"] = tuple([int((i+1) * mult) for i in range(num_hidden)])
        else:
            inverse_layer_num = [1 / (i+1) for i in range(num_hidden)]
            hp_dict["hidden_sizes"] = tuple([int(i * -mult) for i in inverse_layer_num])
    return hp_dict


def run_bo(dataset: BaseDataset):
    def get_response(trial:torch.Tensor)->float:
        with torch.no_grad():
            # convert values to a valid HP setting
            values = to_dict(trial)
            values["disable_progess_bar"] = util.is_cluster
        responses = 0.0
        for _ in range(args.runs_per_trial):
            responses += fit_and_evaluate(dataset,
                                          **values)
        del values
        return responses / args.runs_per_trial


    # MAIN
    # Initial Round
    hp_trials = torch.Tensor([
        [0.001, # learning rate
         0.0,   # weight decay
         64,    # batch size
         3,     # num hidden layers
         20.0,    # hidden size multiplier
         ]
    ]).double()
    hp_trials[0] = normalize(hp_trials[0])

    resp = get_response(hp_trials[0])
    hp_responses = torch.Tensor([[resp]]).double()

    best_performance = resp
    best_trial = to_dict(hp_trials[0])
    # best_trial = [f"{k}: %3.5f" % (v) for k, v in to_dict(hp_trials[0]).items()]

    for i in range(args.trials):
        gp = SingleTaskGP(hp_trials, hp_responses)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        UCB = UpperConfidenceBound(gp, beta=0.1)
        candidate, acq_value = optimize_acqf(UCB, bounds=norm_bounds, q=1, num_restarts=5, raw_samples=20)
        candidate = candidate[0]
        resp = get_response(candidate)

        values = to_dict(candidate)
        # values = [f"{k}: %3.5f" % (v) for k, v in to_dict(candidate).items()]
        if resp > best_performance:
            best_performance = resp
            best_trial = values
        print(f"Run: {i}")
        print(f"tried {values} -> %1.5f"%(resp))
        print(f"best  {best_trial} -> %1.5f"%(best_performance))

        hp_trials = torch.cat([hp_trials, candidate.unsqueeze(0)], dim=0)
        hp_responses = torch.cat([hp_responses, torch.Tensor([[resp]])], dim=0)

    return hp_trials, hp_responses



if __name__ == '__main__':
    args = parser.parse_args()

    from datasets import Cifar10
    dataset = Cifar10(cache_folder="../datasets")
    # dataset = Splice(cache_folder="../datasets")
    dataset = dataset.to(util.device)

    hp_trials, hp_responses = run_bo(dataset)
    y = hp_responses.cpu().numpy()
    result_dict = {}
    result_dict["lr"] = []
    result_dict["weight_decay"] = []
    result_dict["batch_size"]  = []
    result_dict["hidden_sizes"] = []
    result_dict["response"] = []
    for trial, hp_setting in enumerate(hp_trials):
        d = to_dict(hp_setting)
        result_dict["lr"].append(d["lr"])
        result_dict["weight_decay"].append(d["weight_decay"])
        result_dict["batch_size"].append(d["batch_size"])
        result_dict["hidden_sizes"].append(str(d["hidden_sizes"])) # represent tuple as string
        result_dict["response"].append(y[trial])

    df = pd.DataFrame.from_records(result_dict)
    filename = "bo_result.csv"
    folder = os.path.join("runs", str(time.time()))
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, filename)
    df.to_csv(filename)
    # write some meta information
    filename = os.path.join(folder, "meta.txt")
    with open(filename, "w") as f:
        if hasattr(dataset, "get_meta_data"):
            f.write("# Dataset: \n")
            f.write(f"{dataset.get_meta_data()} \n\n")