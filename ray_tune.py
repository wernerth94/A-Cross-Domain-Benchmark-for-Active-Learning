import argparse
import os
import math
from datetime import datetime
from os.path import *
from functools import partial
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from core.helper_functions import get_dataset_by_name, EarlyStopping
from train_encoder import main
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--dataset', type=str, default="dna")
parser.add_argument('--num_trials', type=int, default=100)
parser.add_argument('--max_conc_trials', type=int, default=15)

def evaluate_encoded_classification_config(raytune_config, DatasetClass, config_file, cache_folder, benchmark_folder):
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    hidden_dims = [raytune_config["h1"]]
    if raytune_config["h2"] > 0:
        hidden_dims.append(raytune_config["h2"])
    config["classifier_embedded"]["hidden"] = hidden_dims
    config["dataset_embedded"]["encoder_checkpoint"] = join(benchmark_folder, config["dataset_embedded"]["encoder_checkpoint"])
    config["optimizer_embedded"]["type"] = raytune_config["type"]
    config["optimizer_embedded"]["lr"] = raytune_config["lr"]
    config["optimizer_embedded"]["weight_decay"] = raytune_config["weight_decay"]

    loss_sum = 0.0
    acc_sum = 0.0
    restarts = 3
    for i in range(restarts):
        pool_rng = np.random.default_rng(1)
        model_rng = torch.Generator()
        model_rng.manual_seed(1)
        dataset = DatasetClass(cache_folder, config, pool_rng, encoded=True)
        model = dataset.get_classifier(model_rng)
        loss = nn.CrossEntropyLoss()
        optimizer = dataset.get_optimizer(model)
        batch_size = dataset.classifier_batch_size

        data_loader_rng = torch.Generator()
        data_loader_rng.manual_seed(1)
        train_dataloader = DataLoader(TensorDataset(dataset.x_train, dataset.y_train),
                                      batch_size=batch_size,
                                      generator=data_loader_rng,
                                      shuffle=True, num_workers=2)
        test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=512,
                                     num_workers=4)

        # early_stop = EarlyStopping(patience=40)
        MAX_EPOCHS = 50
        for e in range(MAX_EPOCHS):
            for batch_x, batch_y in train_dataloader:
                yHat = model(batch_x)
                loss_value = loss(yHat, batch_y)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            # early stopping on test
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0
            total = 0.0
            for batch_x, batch_y in test_dataloader:
                yHat = model(batch_x)
                predicted = torch.argmax(yHat, dim=1)
                total += batch_y.size(0)
                test_acc += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
                class_loss = loss(yHat, torch.argmax(batch_y.long(), dim=1))
                test_loss += class_loss.detach().cpu().numpy()
            test_acc /= total
            test_loss /= total
                # if early_stop.check_stop(test_loss):
                #     break
        loss_sum += test_loss
        acc_sum += test_acc
    tune.report(loss=loss_sum/restarts, accuracy=acc_sum/restarts)


def tune_encoded_classification(num_samples, log_folder, config_file, cache_folder, DatasetClass, benchmark_folder):
    log_folder = join(log_folder, "classification_embedded")

    ray_config = {
        "type": tune.choice(["NAdam", "Adam", "SGD"]),
        "lr": tune.loguniform(1e-6, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-3),
        "h1": tune.choice([12, 24, 48]),
        "h2": tune.choice([0, 12, 24, 48]),
    }

    # fixes some parameters of the function
    evaluate_config = partial(evaluate_encoded_classification_config,
                              DatasetClass=DatasetClass,
                              config_file=config_file,
                              cache_folder=cache_folder,
                              benchmark_folder=benchmark_folder)

    analysis = tune.run(evaluate_config,
                        config=ray_config,
                        num_samples=num_samples,
                        metric="loss",
                        mode="min",
                        #scheduler="HyperBand", # Full search might be better for us
                        local_dir=log_folder,
                        max_concurrent_trials=args.max_conc_trials,
                        verbose=1)
    df = analysis.dataframe()
    timestamp = str(datetime.now())[:-7]
    df.to_csv(join(log_folder, f"ray_tune_results_{timestamp}.csv"))



def evaluate_pretext_config(raytune_config, cache_folder, benchmark_folder, dataset):
    # Fake a NameSpace object for the input args
    class FakeNameSpace:
        def __init__(self, df=cache_folder, ds=dataset, seed=1):
            self.data_folder = df
            self.dataset = ds
            self.seed = seed

    # load and modify the config
    with open(join(benchmark_folder, f"configs/{args.dataset}.yaml"), 'r') as f:
        config = yaml.load(f, yaml.Loader)
    hidden_dims = [raytune_config["h1"]]
    if raytune_config["h2"] > 0:
        hidden_dims.append(raytune_config["h2"])
    if raytune_config["h3"] > 0:
        hidden_dims.append(raytune_config["h3"])
    config["pretext_encoder"]["hidden"] = hidden_dims
    config["pretext_encoder"]["feature_dim"] = raytune_config["feature_dim"]
    config["pretext_training"]["batch_size"] = raytune_config["batch_size"]
    config["pretext_optimizer"]["lr"] = raytune_config["lr"]
    config["pretext_optimizer"]["weight_decay"] = raytune_config["weight_decay"]
    config["pretext_optimizer"]["lr_scheduler_decay"] = raytune_config["lr_scheduler_decay"]
    config["pretext_clr_loss"]["temperature"] = raytune_config["temperature"]
    config["pretext_transforms"]["gauss_scale"] = raytune_config["gauss_scale"]
    final_acc = main(FakeNameSpace(), config, store_output=False, verbose=False)
    RESTARTS = 3
    # runs = [main(FakeNameSpace(seed=i), config, store_output=False, verbose=False) for i in range(RESTARTS)]
    # final_acc = sum(runs) / float(RESTARTS)
    tune.report(acc=final_acc)


def tune_pretext(num_samples, cache_folder, benchmark_folder, log_folder, dataset):
    log_folder = join(log_folder, "pretext")
    ray_config = {
        "h1": tune.choice([32, 64, 128]),
        "h2": tune.choice([0, 32, 64, 128]),
        "h3": tune.choice([0, 32, 64, 128]),
        "feature_dim": tune.choice([24, 48]),
        "batch_size": tune.randint(100, 500),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-3),
        "lr_scheduler_decay": tune.loguniform(1e-4, 3e-1),
        "temperature": tune.uniform(0.1, 1.0),
        "gauss_scale": tune.uniform(0.01, 0.3)
    }

    # fixes some parameters of the function
    my_func = partial(evaluate_pretext_config,
                      dataset=dataset,
                      cache_folder=cache_folder,
                      benchmark_folder=benchmark_folder)

    analysis = tune.run(my_func,
                        config=ray_config,
                        num_samples=num_samples,
                        metric="acc",
                        mode="max",
                        #scheduler="HyperBand", # Full search might be better for us
                        local_dir=log_folder,
                        max_concurrent_trials=args.max_conc_trials,
                        verbose=1)
    df = analysis.dataframe()
    timestamp = str(datetime.now())[:-7]
    df.to_csv(join(log_folder, f"ray_tune_results_{timestamp}.csv"))


if __name__ == '__main__':
    args = parser.parse_args()

    benchmark_folder = "al-benchmark"
    base_path = os.path.split(os.getcwd())[0]
    cache_folder = join(base_path, "datasets")

    config_file = join(base_path, benchmark_folder, f"configs/{args.dataset}.yaml")
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)
    # check the dataset
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(cache_folder, config, np.random.default_rng(1), encoded=False)
    # output
    output_folder = join(base_path, benchmark_folder, "raytune_output")
    log_folder = join(output_folder, dataset.name)
    os.makedirs(log_folder, exist_ok=True)

    # tune_pretext(args.num_trials, cache_folder, join(base_path, benchmark_folder), log_folder, args.dataset)
    tune_encoded_classification(args.num_trials, log_folder, config_file, cache_folder, DatasetClass, join(base_path, benchmark_folder))
