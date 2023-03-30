import torch


def get_optimizer_for_dataset(config, model:torch.nn.Module):
    if config["pretext_optimizer"]["type"].lower() == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=config["pretext_optimizer"]["lr"],
                               nesterov=config["pretext_optimizer"]["nesterov"],
                               weight_decay=config["pretext_optimizer"]["weight_decay"],
                               momentum=config["pretext_optimizer"]["momentum"])
    elif config["pretext_optimizer"]["type"].lower() == "nadam":
        return torch.optim.NAdam(model.parameters(),
                                 lr=config["pretext_optimizer"]["lr"],
                                 weight_decay=config["pretext_optimizer"]["weight_decay"])
    else:
        raise NotImplementedError
