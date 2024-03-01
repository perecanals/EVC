import torch

def compute_accuracy(pred, label):
    return torch.mean((pred == label).float()).item()