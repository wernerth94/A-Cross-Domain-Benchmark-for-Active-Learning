import argparse
import os
from datetime import datetime
from os.path import *
from functools import partial
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from core.helper_functions import get_dataset_by_name
from train_encoder import main
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--dataset', type=str, default="splice")

def evaluate_classification_config(config, train_data_file=None, dataset_class=None, cache_folder=None):
    pool_rng = np.random.default_rng(1)
    model_rng = torch.Generator()
    model_rng.manual_seed(1)
    dataset = dataset_class(pool_rng=pool_rng, cache_folder=cache_folder)
    hidden_dims = [config["h1"]]
    if config["h2"] > 0:
        hidden_dims.append(config["h2"])
    if config["h3"] > 0:
        hidden_dims.append(config["h3"])
    model = dataset.get_classifier(model_rng, hidden_dims=hidden_dims)
    loss = nn.CrossEntropyLoss()
    optimizer = dataset.get_optimizer(model, lr=config["lr"], weight_decay=config["weight_decay"])
    batch_size = dataset.classifier_batch_size

    data_loader_rng = torch.Generator()
    data_loader_rng.manual_seed(1)
    train_data = torch.load(train_data_file)
    train_dataloader = DataLoader(TensorDataset(train_data["x_train"], train_data["y_train"]),
                                  batch_size=batch_size,
                                  generator=data_loader_rng,
                                  shuffle=True, num_workers=2)
    test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=512,
                                 num_workers=4)

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
            tune.report(loss=test_loss, accuracy=test_acc)


def tune_classification(num_samples, cache_folder, log_folder, dataset, DatasetClass):
    log_folder = join(log_folder, "classification")

    # data_file = join(base_path, benchmark_folder, "runs/DNA/Oracle/run_1/labeled_data.pt") # oracle data
    data_file = join(cache_folder, dataset.data_file)
    config = {
        "h1": tune.choice([8, 12, 16, 32, 64]),
        "h2": tune.choice([0, 12, 16, 32, 64]),
        "h3": tune.choice([0, 16, 32, 64, 128]),
        "lr": tune.loguniform(1e-6, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-3)
    }

    # fixes some parameters of the function
    evaluate_config = partial(evaluate_classification_config,
                              train_data_file=data_file,
                              dataset_class=DatasetClass,
                              cache_folder=cache_folder)

    analysis = tune.run(evaluate_config,
                        config=config,
                        num_samples=num_samples,
                        metric="loss",
                        mode="min",
                        #scheduler="HyperBand", # Full search might be better for us
                        local_dir=log_folder,
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
    args = FakeNameSpace()
    # load and modify the config
    with open(join(benchmark_folder, f"sim_clr/configs/{args.dataset}.yaml"), 'r') as f:
        config = yaml.load(f, yaml.Loader)
    hidden_dims = [raytune_config["h1"]]
    if raytune_config["h2"] > 0:
        hidden_dims.append(raytune_config["h2"])
    if raytune_config["h3"] > 0:
        hidden_dims.append(raytune_config["h3"])
    config["encoder"]["encoder_hidden"] = hidden_dims
    config["encoder"]["encoder_dim"] = hidden_dims[-1]
    config["encoder"]["feature_dim"] = raytune_config["feature_dim"]
    config["data"]["batch_size"] = raytune_config["batch_size"]
    config["optimizer"]["lr"] = raytune_config["lr"]
    config["optimizer"]["weight_decay"] = raytune_config["weight_decay"]
    config["optimizer"]["lr_scheduler_decay"] = raytune_config["lr_scheduler_decay"]
    final_acc = main(args, config, store_output=False)
    tune.report(acc=final_acc)


def tune_pretext(num_samples, cache_folder, benchmark_folder, log_folder, dataset):
    log_folder = join(log_folder, "pretext")
    ray_config = {
        "h1": tune.choice([8, 12, 16, 32, 64]),
        "h2": tune.choice([0, 12, 16, 32, 64]),
        "h3": tune.choice([0, 16, 32, 64, 128]),
        "feature_dim": tune.choice([8, 12, 24, 48]),
        "batch_size": tune.randint(12, 500),
        "lr": tune.loguniform(1e-6, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-3),
        "lr_scheduler_decay": tune.loguniform(1e-3, 5e-1)
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
                        verbose=1)
    df = analysis.dataframe()
    timestamp = str(datetime.now())[:-7]
    df.to_csv(join(log_folder, f"ray_tune_results_{timestamp}.csv"))


if __name__ == '__main__':
    args = parser.parse_args()
    NUM_SAMPLES = 100
    base_path = os.path.split(os.getcwd())[0]
    cache_folder = join(base_path, "datasets")
    # check the dataset
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(np.random.default_rng(1), cache_folder=cache_folder)
    # output
    benchmark_folder = "al-benchmark"
    output_folder = join(base_path, benchmark_folder, "raytune_output")
    log_folder = join(output_folder, dataset.name)
    os.makedirs(log_folder, exist_ok=True)

    tune_pretext(NUM_SAMPLES, cache_folder, join(base_path, benchmark_folder), log_folder, args.dataset)
    # tune_classification(NUM_SAMPLES, cache_folder, log_folder, dataset, DatasetClass)
