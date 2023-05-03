from os.path import join
from functools import partial
from datetime import datetime
import numpy as np
import yaml
from tqdm import tqdm
from ray import tune
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def evaluate_classification_config(raytune_config, DatasetClass, config_file, cache_folder, benchmark_folder):
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    # hidden_dims = [raytune_config["h1"]]
    # if raytune_config["h2"] > 0:
    #     hidden_dims.append(raytune_config["h2"])
    # config["classifier"]["hidden"] = hidden_dims
    config["optimizer"]["type"] = raytune_config["type"]
    config["optimizer"]["lr"] = raytune_config["lr"]
    config["optimizer"]["weight_decay"] = raytune_config["weight_decay"]
    config["classifier"]["dropout"] = raytune_config["dropout"]

    loss_sum = 0.0
    acc_sum = 0.0
    restarts = 2
    for i in range(restarts):
        pool_rng = np.random.default_rng(1)
        model_rng = torch.Generator()
        model_rng.manual_seed(1)
        dataset = DatasetClass(cache_folder, config, pool_rng, encoded=False)
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
        for e in tqdm(range(MAX_EPOCHS), miniters=1):
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


def tune_classification(num_samples, max_conc_trials, log_folder, config_file, cache_folder, DatasetClass, benchmark_folder):
    log_folder = join(log_folder, "classification")

    ray_config = {
        "type": tune.choice(["NAdam", "Adam", "SGD"]),
        "lr": tune.loguniform(1e-6, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-3),
        "dropout": tune.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        # "h1": tune.choice([12, 24, 48]),
        # "h2": tune.choice([0, 12, 24, 48]),
    }

    # fixes some parameters of the function
    evaluate_config = partial(evaluate_classification_config,
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
                        max_concurrent_trials=max_conc_trials,
                        verbose=1)
    df = analysis.dataframe()
    timestamp = str(datetime.now())[:-7]
    df.to_csv(join(log_folder, f"ray_tune_results_{timestamp}.csv"))

