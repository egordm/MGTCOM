import re
from typing import Dict, List, Iterable, Tuple

import pandas as pd

EdgeList = pd.DataFrame
ComList = pd.DataFrame
Coms = Dict[int, List[int]]


def read_edgelist(filepath: str) -> EdgeList:
    with open(filepath, 'r') as f:
        return pd.DataFrame([
            tuple(map(int, re.split(r'\s|\t', line.strip())))[:2]
            for line in f.readlines()
        ], columns=['src', 'dst'])


def read_comlist(filepath: str, delimiter='\t') -> ComList:
    return pd.read_csv(filepath, sep=delimiter, header=None, names=['nid', 'cid'])


def write_comlist(comlist: ComList, filepath: str):
    comlist.to_csv(filepath, sep='\t', index=False, header=False, columns=['nid', 'cid'])


def read_coms(filepath: str) -> Coms:
    with open(filepath, 'r') as f:
        return {
            (i + 1): [int(x) for x in re.split(r'\s|\t', line.strip())]
            for i, line in enumerate(f.readlines())
        }


def write_coms(coms: Coms, filepath: str):
    with open(filepath, 'w') as f:
        for _, coms in sorted(coms.items()):
            f.write(' '.join(coms))


def coms_to_comlist(coms: Coms) -> ComList:
    return pd.DataFrame([
        (nid, cid)
        for cid, nids in coms.items()
        for nid in nids
    ], columns=['nid', 'cid'])


def comlist_to_coms(comlist: ComList) -> Coms:
    raise NotImplementedError()
    # return {
    #     cid: nids
    #     for row in comlist.groupby('cid').apply(lambda x: x.nid.tolist()).iterrows()
    # }


def write_edgelist(edges: Iterable[Tuple[int, int]], f):
    # TODO: change usages to use the above type definition
    for edge in sorted(edges):
        f.write('{}\t{}\n'.format(*edge))
