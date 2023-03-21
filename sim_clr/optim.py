import torch


def get_optimizer_for_dataset(dataset_name, model:torch.nn.Module):
    if dataset_name == "cifar10":
        return torch.optim.SGD(model.parameters(), lr=0.4, nesterov=False, weight_decay=0.0001, momentum=0.9)
    else:
        raise NotImplementedError
