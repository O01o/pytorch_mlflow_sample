import torch
import torch.nn as nn


def get_variance(loss: nn.CrossEntropyLoss):
    batch_losses = loss.detach().cpu().flatten()
    return torch.var(batch_losses)