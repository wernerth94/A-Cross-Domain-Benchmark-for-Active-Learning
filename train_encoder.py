"""
Based on code from TypiClust: https://github.com/avihu111/typiclust
Hacohen, Guy, Avihu Dekel, and Daphna Weinshall.
"Active learning on a budget: Opposite strategies suit high and low budgets." arXiv preprint arXiv:2202.02794 (2022).
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import experiment_util as util
import argparse
import os
import torch
import numpy as np
from sim_clr.data import get_raw_data_by_name, \
                         get_train_dataloader_for_dataset, get_validation_dataloader_for_dataset, \
                         get_transforms_for_dataset, get_validation_transforms_for_dataset, AugmentedDataset
from sim_clr.encoder import get_encoder_for_dataset
from sim_clr.memory import create_memory_bank
from sim_clr.loss import get_loss_for_dataset
from sim_clr.optim import get_optimizer_for_dataset
from sim_clr.training import get_training_parameters, adjust_learning_rate, simclr_train, fill_memory_bank
from sim_clr.evaluate import contrastive_evaluate

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

def main():


    # Model
    print('Retrieve model')
    model = get_encoder_for_dataset(args.dataset)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    #print(model)
    model = model.to(util.device)

    # CUDNN
    print('Set CuDNN benchmark')
    torch.backends.cudnn.benchmark = True

    # Dataset
    print('Retrieve dataset')
    train_dataset, val_dataset = get_raw_data_by_name(args.data_folder, args.dataset)

    train_dataset.transform = get_transforms_for_dataset(args.dataset)
    val_dataset.transform = get_validation_transforms_for_dataset(args.dataset)
    train_dataset = AugmentedDataset(train_dataset)
    val_dataset = AugmentedDataset(val_dataset)

    train_dataloader = get_train_dataloader_for_dataset(args.dataset, train_dataset)
    val_dataloader = get_validation_dataloader_for_dataset(args.dataset, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print('Build MemoryBank')
    base_dataset, _ = get_raw_data_by_name(args.data_folder, args.dataset)
    base_dataset.transform = get_validation_transforms_for_dataset(args.dataset) # Dataset w/o augs for knn eval
    base_dataset = AugmentedDataset(base_dataset)
    base_dataloader = get_validation_dataloader_for_dataset(args.dataset, base_dataset)

    memory_bank_base = create_memory_bank(args.dataset, base_dataset)
    memory_bank_base = memory_bank_base.to(util.device)
    memory_bank_val = create_memory_bank(args.dataset, val_dataset)
    memory_bank_val = memory_bank_val.to(util.device)

    # Criterion
    print('Retrieve criterion')
    criterion = get_loss_for_dataset(args.dataset, util.device)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(util.device)

    # Optimizer and scheduler
    print('Retrieve optimizer')
    optimizer = get_optimizer_for_dataset(args.dataset, model)
    print(optimizer)

    pretext_checkpoint = os.path.join("checkpoints", args.dataset, f'checkpoint_seed{args.seed}.pth.tar')
    pretext_model = os.path.join("checkpoints", args.dataset, f'model_seed{args.seed}.pth.tar')
    pretext_features = os.path.join("checkpoints", args.dataset, f'features_seed{args.seed}.npy')
    topk_neighbors_train_path = os.path.join("checkpoints", args.dataset, f'topk-train-neighbors_seed{args.seed}.npy')
    topk_neighbors_val_path = os.path.join("checkpoints", args.dataset, f'topk-val-neighbors_seed{args.seed}.npy')

    start_epoch = 0
    # Checkpoint
    # if os.path.exists(config['pretext_checkpoint']):
    #     print('Restart from checkpoint {}'.format(config['pretext_checkpoint']))
    #     checkpoint = torch.load(config['pretext_checkpoint'], map_location='cpu')
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     model.load_state_dict(checkpoint['model'])
    #     model.cuda()
    #     start_epoch = checkpoint['epoch']
    # else:
    #     print('No checkpoint file at {}'.format(config['pretext_checkpoint']))
    #     start_epoch = 0
    #     model = model.cuda()

    epochs = get_training_parameters(args.dataset)
    # Training
    print('Starting main loop')
    for epoch in range(start_epoch, epochs):
        print('Epoch %d/%d' %(epoch, epochs))
        print('-'*15)

        # Adjust lr
        lr = adjust_learning_rate(args.dataset, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch, util.device)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' %(top1))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, pretext_checkpoint)

        topk = 20
        print('Mine the nearest neighbors (Top-%d)' % (topk))
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        np.save(topk_neighbors_train_path, indices)
        np.save(pretext_features, memory_bank_base.pre_lasts.cpu().numpy())
        np.save(pretext_features.replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())

    # Save final model
    torch.save(model.state_dict(), pretext_model)

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    print('Fill memory bank for mining the nearest neighbors (train) ...')
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(topk_neighbors_train_path, indices)
    # save features
    np.save(pretext_features, memory_bank_base.pre_lasts.cpu().numpy())
    np.save(pretext_features.replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())


    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print('Fill memory bank for mining the nearest neighbors (val) ...')
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(topk_neighbors_val_path, indices)


if __name__ == '__main__':
    main()
