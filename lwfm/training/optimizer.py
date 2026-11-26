import torch


def build_optimizer(prameters, config):
    optimizer_registry = {"sgd": torch.optim.SGD}

    optimizer_name = config["name"]
    learning_rate = config["lr"]

    if optimizer_name not in optimizer_name:
        raise ValueError(f"Unknown Optimizer: {config['name']}")

    optimizer = optimizer_registry[optimizer_name](
        prameters, lr=learning_rate, **config.get("kwargs", {})
    )

    return optimizer


def build_scheduler(config, optimizer, epochs, train_len):
    scheduler_registry = {
        "cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR
    }

    scheduler_name = config["name"]

    if scheduler_name not in scheduler_registry:
        raise ValueError(f"Unknown Scheduler: {scheduler_name}")

    scheduler = scheduler_registry[scheduler_name](
        optimizer, T_max=epochs * train_len, **config.get("kwargs", {})
    )

    return scheduler
