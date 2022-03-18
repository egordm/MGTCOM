import torch


def pairwise_dotp(x, y):
    return torch.einsum('...i,...i->...', x, y)


def pairwise_euclidean(x, y, p=2):
    return torch.cdist(x, y, p=p)


def pairwise_euclidean_sim(x, y, p=2):
    return -torch.cdist(x, y, p=p)


def pairwise_cosine(x, y):
    return torch.cosine_similarity(x, y, dim=-1)
