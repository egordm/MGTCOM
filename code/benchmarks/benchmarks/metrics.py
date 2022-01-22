from sklearn.metrics import normalized_mutual_info_score

from datasets.formats import LabelList


def nmi(x: LabelList, y: LabelList) -> float:
    assert_labellists_aligned(x, y)

    return normalized_mutual_info_score(x, y)


def assert_labellists_aligned(x: LabelList, y: LabelList):
    if (x['nid'] != y['nid']).any():
        x.sort_values('nid', inplace=True)
        y.sort_values('nid', inplace=True)

        if (x['nid'] != y['nid']).any():
            raise ValueError('x and y must have the same nid')