from nf1 import NF1
from sklearn.metrics import normalized_mutual_info_score

from datasets.formats import ComList, Coms


def nmi(x: ComList, y: ComList) -> float:
    """
    Compute the normalized mutual information score of two communities.
    The measure is community label invariant.
    :param x:
    :param y:
    :return:
    """
    assert_comlists_aligned(x, y)

    return normalized_mutual_info_score(x['cid'].values, y['cid'].values)


def nf1(x: Coms, y: Coms) -> float:
    """
    Compute the normalized F1 score of two communities.
    The measure is community and node order invariant.
    :param x:
    :param y:
    :return:
    """
    nf = NF1(list(x.values()), list(y.values()))
    summary = nf.summary()
    scores, details = summary['scores'], summary['details']

    return float(scores.loc['NF1'])


def assert_comlists_aligned(x: ComList, y: ComList):
    """
    Assert that the two comlists are aligned (same length and same node ids).
    :param x:
    :param y:
    """
    if (x['nid'] != y['nid']).any():
        x.sort_values('nid', inplace=True)
        y.sort_values('nid', inplace=True)

        if (x['nid'] != y['nid']).any():
            raise ValueError('x and y must have the same nid')
