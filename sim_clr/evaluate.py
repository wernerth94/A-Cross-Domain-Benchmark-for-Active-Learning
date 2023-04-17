from training import AverageMeter
import torch
import torch.nn as nn
from core.helper_functions import EarlyStopping

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


def linear_evaluate(train_loader, val_loader, model, embedding_dim, n_classes, device):
    model.eval()
    class_head = nn.Linear(embedding_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(class_head.parameters(), lr=0.05)
    loss = nn.CrossEntropyLoss().to(device)
    max_acc = 0.0
    all_accs = []
    early_stop = EarlyStopping(patience=5, lower_is_better=False)
    for epoch in range(30):
        for batch in train_loader:
            x = batch['image'].to(device)
            y = batch['target'].to(device)
            with torch.no_grad():
                x_enc = model(x).detach()
            y_hat = class_head(x_enc)
            loss_value = loss(y_hat, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        # validation
        with torch.no_grad():
            total = 0.0
            correct = 0.0
            for batch in val_loader:
                batch_x, batch_y = batch["image"].to(device), batch["target"].to(device)
                x_enc = model(batch_x)
                y_hat = class_head(x_enc)
                predicted = torch.argmax(y_hat, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                # class_loss = self.loss(y_hat, torch.argmax(batch_y.long(), dim=1))
                # loss_sum += class_loss.detach().cpu().numpy()
            accuracy = correct / total
            all_accs.append(accuracy)
            # early stop on test with patience of 0
            if early_stop.check_stop(accuracy):
                # print("early stop", len(all_accs))
                # print(["%1.4f"%(a) for a in all_accs])
                return accuracy * 100.0
            max_acc = max(accuracy, max_acc)
    # print(["%1.4f"%(a) for a in all_accs[:10]])
    # print(["%1.4f"%(a) for a in all_accs[-10:]])
    return accuracy * 100.0
