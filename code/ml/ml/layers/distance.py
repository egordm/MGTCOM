import torch


def pairwise_dotp(x, y):
    return torch.einsum('...i,...i->...', x, y)


def pairwise_euclidean(x, y, p=2):
    return torch.cdist(x, y, p=p)


def pairwise_euclidean_sim(x, y, p=2):
    return -torch.cdist(x, y, p=p)


def pairwise_cosine(x, y, dim=None):
    if not dim:
        dim = x.dim() - 1
    return torch.cosine_similarity(x.unsqueeze(dim), y.unsqueeze(dim - 1), dim=dim + 1)
