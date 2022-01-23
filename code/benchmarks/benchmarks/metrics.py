import igraph
import numpy as np
from nf1 import NF1
from sklearn.metrics import normalized_mutual_info_score

from shared.graph import ComList, Coms


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

    In order to avoid biased evaluations be sure to comply with the following rules:
        * A community must be composed by at least 3 nodes.
        * No nested communities: a community must not be a proper subset of another community.

    :param x:
    :param y:
    :return:
    """
    nf = NF1(list(x.values()), list(y.values()))
    summary = nf.summary()
    scores, details = summary['scores'], summary['details']

    return float(scores.loc['NF1'])


def modularity(G: igraph.Graph, coms: ComList) -> float:
    """
    Compute the modularity of a community.
    :param G:
    :param coms:
    :return:
    """
    return G.modularity(coms['cid'].values)


def conductance(G: igraph.Graph, coms: Coms) -> float:
    """
    Compute the conductance of a community.

    Fraction of total edge volume that points outside the community.
    .. math:: f(S) = \\frac{c_S}{2 m_S+c_S}
    where :math:`c_S` is the number of community nodes and, :math:`m_S` is the number of community edges

    Source: https://github.com/GiulioRossetti/cdlib
    :param G:
    :param coms:
    :return:
    """
    values = []
    for com in coms.values():
        a = G.vs[com]
        b = G.vs[set(i for i in range(G.vcount())).difference(com)]
        edges_outside = len(G.es.select(_between=(a, b)))
        ms = len(G.es.select(_between=(a, a)))

        try:
            ratio = float(edges_outside) / ((2 * ms) + edges_outside)
        except:
            ratio = 0
        values.append(ratio)

    return np.mean(values)


def assert_comlists_aligned(x: ComList, y: ComList):
    """
    Assert that the two comlists are aligned (same length and same node ids).
    :param x:
    :param y:
    """
    if (x.index != y.index).any():
        x.sort_index(inplace=True)
        y.sort_index(inplace=True)

        if (x.index != y.index).any():
            raise ValueError('x and y must have the same nid')
