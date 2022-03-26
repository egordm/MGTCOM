import torch


def pairwise_dotp(x, y):
    return torch.einsum('...i,...i->...', x, y)


def pairwise_euclidean(x, y, p=2):
    return torch.cdist(x, y, p=p)


def pairwise_euclidean_sim(x, y, p=2):
    return -torch.cdist(x, y, p=p)


def pairwise_cosine(x, y):
    return torch.cosine_similarity(x, y, dim=-1)


def pairwise_sim_fn(sim='dotp'):
    if sim == 'dotp':
        return pairwise_dotp
    elif sim == 'cosine':
        return pairwise_cosine
    elif sim == 'euclidean':
        return pairwise_euclidean_sim
    else:
        raise ValueError(f'Unknown similarity function {sim}')
