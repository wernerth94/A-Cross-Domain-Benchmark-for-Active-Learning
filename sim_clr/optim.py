import torch


def get_optimizer_for_dataset(config, model:torch.nn.Module):
    if config["optimizer"]["type"].lower() == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=config["optimizer"]["lr"],
                               nesterov=config["optimizer"]["nesterov"],
                               weight_decay=config["optimizer"]["weight_decay"],
                               momentum=config["optimizer"]["momentum"])
    elif config["optimizer"]["type"].lower() == "nadam":
        return torch.optim.NAdam(model.parameters(),
                                 lr=config["optimizer"]["lr"],
                                 weight_decay=config["optimizer"]["weight_decay"])
    else:
        raise NotImplementedError
