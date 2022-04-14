from enum import Enum

import faiss
import torch


def pairwise_dotp(x, y):
    return torch.einsum('...i,...i->...', x, y)


def pairwise_dotp_dist(x, y):
    return -torch.einsum('...i,...i->...', x, y)


def pairwise_l2(x, y):
    return torch.pairwise_distance(x, y, p=2)


def pairwise_l2_sim(x, y):
    return -torch.pairwise_distance(x, y, p=2)


def pairwise_cosine(x, y):
    return torch.cosine_similarity(x, y, dim=-1)


def pairwise_cosine_dist(x, y):
    return 1 - torch.cosine_similarity(x, y, dim=-1)


class Metric(Enum):
    L2 = 'l2'
    COSINE = 'cosine'
    DOTP = 'dotp'

    @property
    def pairwise_sim_fn(self):
        if self == Metric.L2:
            return pairwise_l2_sim
        elif self == Metric.COSINE:
            return pairwise_cosine_dist
        elif self == Metric.DOTP:
            return pairwise_dotp_dist
        else:
            raise ValueError('Unknown metric')

    @property
    def pairwise_dist_fn(self):
        if self == Metric.L2:
            return pairwise_l2
        elif self == Metric.COSINE:
            return pairwise_cosine
        elif self == Metric.DOTP:
            return pairwise_dotp

    def faiss_metric(self):
        if self == Metric.L2:
            return faiss.METRIC_L2
        elif self == Metric.COSINE:
            return faiss.METRIC_INNER_PRODUCT
        elif self == Metric.DOTP:
            return faiss.METRIC_INNER_PRODUCT

    def sk_metric(self) -> str:
        if self == Metric.L2:
            return 'euclidean'
        elif self == Metric.COSINE:
            return 'cosine'
        elif self == Metric.DOTP:
            return 'cosine'


def pairwise_sim_fn(sim='dotp'):
    if sim == 'dotp':
        return pairwise_dotp
    elif sim == 'cosine':
        return pairwise_cosine
    elif sim == 'euclidean':
        return pairwise_l2_sim
    else:
        raise ValueError(f'Unknown similarity function {sim}')


def pairwise_dist_fn(sim='dotp'):
    if sim == 'dotp':
        return pairwise_dotp_dist
    elif sim == 'cosine':
        return pairwise_cosine_dist
    elif sim == 'euclidean':
        return pairwise_l2
    else:
        raise ValueError(f'Unknown distance function {sim}')
