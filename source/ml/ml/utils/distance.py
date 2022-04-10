import torch


def pairwise_dotp(x, y):
    return torch.einsum('...i,...i->...', x, y)


def pairwise_l2(x, y, p=2):
    return torch.pairwise_distance(x, y, p=p)


def pairwise_l2s(x, y, p=2):
    return -torch.pairwise_distance(x, y, p=p)


def pairwise_cosine(x, y):
    return torch.cosine_similarity(x, y, dim=-1)


def pairwise_sim_fn(sim='dotp'):
    if sim == 'dotp':
        return pairwise_dotp
    elif sim == 'cosine':
        return pairwise_cosine
    elif sim == 'euclidean':
        return pairwise_l2s
    else:
        raise ValueError(f'Unknown similarity function {sim}')