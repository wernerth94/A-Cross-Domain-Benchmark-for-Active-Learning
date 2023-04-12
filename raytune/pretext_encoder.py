from os.path import join
from functools import partial
from datetime import datetime
import yaml
from train_encoder import main
from ray import tune

def evaluate_pretext_config(raytune_config, cache_folder, benchmark_folder, dataset_name):
    # Fake a NameSpace object for the input args
    class FakeNameSpace:
        def __init__(self, df=cache_folder, ds=dataset_name, seed=1):
            self.data_folder = df
            self.dataset = ds
            self.seed = seed

    # load and modify the config
    with open(join(benchmark_folder, f"configs/{dataset_name}.yaml"), 'r') as f:
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


def tune_pretext(num_samples, max_conc_trials, cache_folder, benchmark_folder, log_folder, dataset_name):
    log_folder = join(log_folder, "pretext")
    ray_config = {
        "h1": tune.choice([32, 64, 128, 256]),
        "h2": tune.choice([0, 32, 64, 128, 256]),
        "h3": tune.choice([0, 32, 64, 128, 256]),
        "feature_dim": tune.choice([24, 48]),
        "batch_size": tune.randint(100, 500),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-3),
        "lr_scheduler_decay": tune.loguniform(1e-4, 3e-1),
        "temperature": tune.uniform(0.05, 1.0),
        "gauss_scale": tune.uniform(0.01, 0.3)
    }

    # fixes some parameters of the function
    my_func = partial(evaluate_pretext_config,
                      dataset_name=dataset_name,
                      cache_folder=cache_folder,
                      benchmark_folder=benchmark_folder)

    analysis = tune.run(my_func,
                        config=ray_config,
                        num_samples=num_samples,
                        metric="acc",
                        mode="max",
                        #scheduler="HyperBand", # Full search might be better for us
                        local_dir=log_folder,
                        max_concurrent_trials=max_conc_trials,
                        verbose=1)
    df = analysis.dataframe()
    timestamp = str(datetime.now())[:-7]
    df.to_csv(join(log_folder, f"ray_tune_results_{timestamp}.csv"))
