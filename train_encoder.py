"""
Based on code from TypiClust: https://github.com/avihu111/typiclust
Hacohen, Guy, Avihu Dekel, and Daphna Weinshall.
"Active learning on a budget: Opposite strategies suit high and low budgets." arXiv preprint arXiv:2202.02794 (2022).
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import shutil
import yaml
import experiment_util as util
import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from core.helper_functions import get_dataset_by_name
from sim_clr.data import get_train_dataloader_for_dataset, \
                         get_validation_dataloader_for_dataset, \
                         AugmentedDataset
from sim_clr.memory import create_memory_bank
from sim_clr.loss import get_loss_for_dataset
from sim_clr.optim import get_optimizer_for_dataset
from sim_clr.training import adjust_learning_rate, simclr_train, fill_memory_bank
from sim_clr.evaluate import contrastive_evaluate, linear_evaluate

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--dataset', type=str, default="splice")
parser.add_argument('--seed', type=int, default=1)


def main(args, config, store_output=True, verbose=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(args.data_folder, config, np.random.default_rng(args.seed), encoded=False)
    config["n_classes"] = dataset.n_classes
    # Model
    model = dataset.get_pretext_encoder(config, seed=args.seed)
    if verbose:
        print('Retrieve model')
        print('Model is {}'.format(model.__class__.__name__))
        print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    #print(model)
    model = model.to(util.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Dataset
    train_dataset, val_dataset = dataset.load_pretext_data()

    train_dataset.transform = dataset.get_pretext_transforms(config)
    val_dataset.transform = dataset.get_pretext_validation_transforms(config)
    train_dataset = AugmentedDataset(train_dataset)
    val_dataset = AugmentedDataset(val_dataset)

    train_dataloader = get_train_dataloader_for_dataset(config, train_dataset)
    val_dataloader = get_validation_dataloader_for_dataset(config, val_dataset)
    if verbose:
        print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    base_dataset, _ = dataset.load_pretext_data()
    base_dataset.transform = dataset.get_pretext_validation_transforms(config) # Dataset w/o augs for knn eval
    base_dataset = AugmentedDataset(base_dataset)
    base_dataloader = get_validation_dataloader_for_dataset(config, base_dataset)

    memory_bank_base = create_memory_bank(config, base_dataset)
    memory_bank_base.to(util.device)
    memory_bank_val = create_memory_bank(config, val_dataset)
    memory_bank_val.to(util.device)

    # Criterion
    criterion = get_loss_for_dataset(config, util.device)
    if verbose:
        print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(util.device)

    # Optimizer and scheduler
    optimizer = get_optimizer_for_dataset(config, model)
    print(optimizer)

    chkpt_folder = os.path.join("encoder_checkpoints", args.dataset)
    if store_output:
        if os.path.exists(chkpt_folder):
            shutil.rmtree(chkpt_folder)
        os.makedirs(chkpt_folder, exist_ok=True)
    pretext_model = os.path.join(chkpt_folder, f'model_seed{args.seed}.pth.tar')

    if store_output:
        writer = SummaryWriter(chkpt_folder)

    moving_avrg = 0.0
    start_epoch = 0
    epochs = config["pretext_training"]["epochs"]
    # Training
    for epoch in range(start_epoch, epochs):

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        if verbose:
            print('Epoch %d/%d' %(epoch, epochs))
            print('-'*15)
            print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        simclr_train(train_dataloader, model, criterion, optimizer, epoch, util.device)

        # # Fill memory bank
        # fill_memory_bank(base_dataloader, model, memory_bank_base, util.device)
        # # Evaluate (To monitor progress - Not for validation)
        # top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base, util.device)

        top1 = linear_evaluate(base_dataloader, val_dataloader, model,
                               config["pretext_encoder"]["feature_dim"], dataset.n_classes, util.device)


        model.train()
        moving_avrg = 0.9 * moving_avrg + 0.1 * top1
        if store_output:
            writer.add_scalar("kNN Eval", top1, epoch)
        if verbose:
            print('Result of kNN evaluation is %.2f' %(top1))

    if store_output:
        # End logging
        writer.close()

    if store_output:
        # Save final model
        torch.save(model.state_dict(), pretext_model)
    return moving_avrg


if __name__ == '__main__':
    args = parser.parse_args()
    with open(f"configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    main(args, config)
