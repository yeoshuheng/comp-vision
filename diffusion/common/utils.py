import torch

def unsqueeze_to(tensor, ndim : int):
    while (tensor.ndim < ndim):
        tensor = tensor.unsequeeze(-1)
    return tensor

def unsqueeze_as(tensor, target_tensor):
    while tensor.ndim < target_tensor.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor