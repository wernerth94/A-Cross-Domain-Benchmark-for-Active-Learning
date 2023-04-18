"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math
import torch

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def get_training_parameters(dataset_name):
    if dataset_name == "cifar10":
        return 500
    if dataset_name == "splice":
        return 200
    else:
        raise NotImplementedError


def adjust_learning_rate(config, optimizer, epoch):

    def cosine(lr, decay, max_epochs):
        eta_min = lr * (decay ** 3)
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2

    if config["pretext_optimizer"]["lr_scheduler"] == "cosine":
        lr = cosine(config["pretext_optimizer"]["lr"],
                    config["pretext_optimizer"]["lr_scheduler_decay"],
                    config["pretext_training"]["epochs"])
    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def simclr_train(train_loader, model, criterion, optimizer, epoch, device):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        if len(images.size()) == 4:
            b, c, h, w = images.size()
            input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
        elif len(images.size()) == 2:
            input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, images.size(-1))
        else:
            raise NotImplementedError
        input_ = input_.to(device, non_blocking=True)
        # targets = batch['target'].to(device, non_blocking=True)

        output = model(input_).view(images.size(0), 2, -1)
        loss = criterion(output)
        total_loss += loss.detach().item()
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
    return total_loss

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, device):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        output, pre_last = model(images, return_pre_last=True)
        memory_bank.update(output, pre_last, targets)
        # if i % 100 == 0:
        #     print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
