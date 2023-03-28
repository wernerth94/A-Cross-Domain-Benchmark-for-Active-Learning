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
from sim_clr.training import get_training_parameters, adjust_learning_rate, simclr_train, fill_memory_bank
from sim_clr.evaluate import contrastive_evaluate

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--dataset', type=str, default="splice")
parser.add_argument('--seed', type=int, default=1)


def main(args):
    with open(f"sim_clr/configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(np.random.default_rng(args.seed), cache_folder=args.data_folder)
    # Model
    print('Retrieve model')
    model = dataset.get_pretext_encoder(config, seed=args.seed)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    #print(model)
    model = model.to(util.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Dataset
    print('Retrieve dataset')
    train_dataset, val_dataset = dataset.load_pretext_data()

    train_dataset.transform = dataset.get_pretext_transforms()
    val_dataset.transform = dataset.get_pretext_validation_transforms()
    train_dataset = AugmentedDataset(train_dataset)
    val_dataset = AugmentedDataset(val_dataset)

    train_dataloader = get_train_dataloader_for_dataset(config, train_dataset)
    val_dataloader = get_validation_dataloader_for_dataset(config, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print('Build MemoryBank')
    base_dataset, _ = dataset.load_pretext_data()
    base_dataset.transform = dataset.get_pretext_validation_transforms() # Dataset w/o augs for knn eval
    base_dataset = AugmentedDataset(base_dataset)
    base_dataloader = get_validation_dataloader_for_dataset(config, base_dataset)

    memory_bank_base = create_memory_bank(config, base_dataset)
    memory_bank_base.to(util.device)
    memory_bank_val = create_memory_bank(config, val_dataset)
    memory_bank_val.to(util.device)

    # Criterion
    print('Retrieve criterion')
    criterion = get_loss_for_dataset(config, util.device)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(util.device)

    # Optimizer and scheduler
    print('Retrieve optimizer')
    optimizer = get_optimizer_for_dataset(config, model)
    print(optimizer)

    chkpt_folder = os.path.join("encoder_checkpoints", args.dataset)
    if os.path.exists(chkpt_folder):
        shutil.rmtree(chkpt_folder)
    os.makedirs(chkpt_folder, exist_ok=True)
    pretext_checkpoint = os.path.join(chkpt_folder, f'checkpoint_seed{args.seed}.pth.tar')
    pretext_model = os.path.join(chkpt_folder, f'model_seed{args.seed}.pth.tar')
    pretext_features = os.path.join(chkpt_folder, f'features_seed{args.seed}.npy')
    topk_neighbors_train_path = os.path.join(chkpt_folder, f'topk-train-neighbors_seed{args.seed}.npy')
    topk_neighbors_val_path = os.path.join(chkpt_folder, f'topk-val-neighbors_seed{args.seed}.npy')

    writer = SummaryWriter(chkpt_folder)

    moving_avrg = 0.0
    start_epoch = 0
    epochs = config["training"]["epochs"]
    # Training
    print('Starting main loop')
    for epoch in range(start_epoch, epochs):
        print('Epoch %d/%d' %(epoch, epochs))
        print('-'*15)

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch, util.device)

        # Fill memory bank
        fill_memory_bank(base_dataloader, model, memory_bank_base, util.device)

        # Evaluate (To monitor progress - Not for validation)
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base, util.device)
        moving_avrg = 0.9 * moving_avrg + 0.1 * top1
        writer.add_scalar("kNN Eval", top1, epoch)
        print('Result of kNN evaluation is %.2f' %(top1))

        # Checkpoint
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, pretext_checkpoint)

        # topk = 20
        # print('Mine the nearest neighbors (Top-%d)' % (topk))
        # indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        # np.save(topk_neighbors_train_path, indices)
        # np.save(pretext_features, memory_bank_base.pre_lasts.cpu().numpy())
        # np.save(pretext_features.replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())

    # End logging
    writer.close()

    # Save final model
    torch.save(model.state_dict(), pretext_model)
    return moving_avrg
    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    # print('Fill memory bank for mining the nearest neighbors (train) ...')
    # fill_memory_bank(base_dataloader, model, memory_bank_base)
    # topk = 20
    # print('Mine the nearest neighbors (Top-%d)' %(topk))
    # indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    # np.save(topk_neighbors_train_path, indices)
    # # save features
    # np.save(pretext_features, memory_bank_base.pre_lasts.cpu().numpy())
    # np.save(pretext_features.replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())


    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    # print('Fill memory bank for mining the nearest neighbors (val) ...')
    # fill_memory_bank(val_dataloader, model, memory_bank_val)
    # topk = 5
    # print('Mine the nearest neighbors (Top-%d)' %(topk))
    # indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    # np.save(topk_neighbors_val_path, indices)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
