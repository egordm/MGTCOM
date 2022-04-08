from collections import defaultdict


def merge_dicts(ds, merge_fn=None):
    res = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            res[k].append(v)

    if merge_fn is not None:
        for k, v in res.items():
            res[k] = merge_fn(v)

    return res