import argparse
import os
from os.path import *
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from core.helper_functions import get_dataset_by_name
from ray import tune

parser = argparse.ArgumentParser()

def evaluate_config(config, train_data_file=None, dataset_class=None, cache_folder=None):
    dataset = dataset_class(cache_folder=cache_folder)
    hidden_dims = [config["h1"]]
    if config["h2"] > 0:
        hidden_dims.append(config["h2"])
    if config["h3"] > 0:
        hidden_dims.append(config["h3"])
    model = dataset.get_classifier(hidden_dims=hidden_dims)
    loss = nn.CrossEntropyLoss()
    optimizer = dataset.get_optimizer(model, lr=config["lr"], weight_decay=config["weight_decay"])
    batch_size = dataset.classifier_batch_size

    train_data = torch.load(train_data_file)
    train_dataloader = DataLoader(TensorDataset(train_data["x_train"], train_data["y_train"]),
                                  batch_size=batch_size,
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


if __name__ == '__main__':
    args = parser.parse_args()
    NUM_SAMPLES = 100
    base_path = os.path.split(os.getcwd())[0]
    if exists(join(base_path, "al_benchmark")):
        benchmark_folder = "al_benchmark"
    elif exists(join(base_path, "al-benchmark")):
        benchmark_folder = "al-benchmark"
    else:
        raise NameError(f"Benchmark folder cannot be fount in {base_path}")
        # output
    log_folder = join(base_path, benchmark_folder, "runs/Splice")
    # test data loading from the original dataset
    cache_folder = join(base_path, "datasets")
    # training data
    # all training data
    data_file = join(cache_folder, "splice_al.pt")
    # oracle data
    # data_file = join(base_path, benchmark_folder, "runs/DNA/Oracle/run_1/labeled_data.pt")
    dataset_class = get_dataset_by_name("splice")
    config = {
        "h1": tune.choice([8, 12, 16, 32, 64]),
        "h2": tune.choice([0, 12, 16, 32, 64]),
        "h3": tune.choice([0, 16, 32, 64, 128]),
        "lr": tune.loguniform(1e-6, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-3)
    }

    # fixes some parameters of the function
    evaluate_config = partial(evaluate_config,
                              train_data_file=data_file,
                              dataset_class=dataset_class,
                              cache_folder=cache_folder)

    analysis = tune.run(evaluate_config,
                        config=config,
                        num_samples=NUM_SAMPLES,
                        metric="loss",
                        mode="min",
                        #scheduler="HyperBand", # Full search might be better for us
                        local_dir=log_folder,
                        verbose=1)
    df = analysis.dataframe()
    df.to_csv(join(log_folder, "ray_tune_results.csv"))
