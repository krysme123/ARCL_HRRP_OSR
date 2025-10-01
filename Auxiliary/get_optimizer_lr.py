import torch

def get_optimizer_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()
