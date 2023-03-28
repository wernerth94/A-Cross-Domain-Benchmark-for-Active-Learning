from training import AverageMeter
import torch

@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank, device):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output)

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg
