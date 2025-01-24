import torch


# TODO JL 1/22/25: Add NNCG and Shampoo optimizers
def get_optimizer(model, optimizer_config):

    optimizer = getattr(torch.optim, optimizer_config.name)(
        model.parameters(),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.wd,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
    )

    return optimizer
