import re
from typing import Dict, List

import igraph as ig
import pandas as pd

EdgeList = pd.DataFrame
ComList = pd.DataFrame
Coms = Dict[int, List[int]]


def read_comlist(filepath: str, delimiter='\t', named=False) -> ComList:
    """
    Reads a community lists from a file.

    comlist specification:
    - no header
    - two columns, separated by a tab
    - first column is the node index
    - second column is the community index
    - all indices should be integers
    - all indices should be 1-indexed

    if named is True, then numeric index contraints are ignored

    :param filepath:
    :param delimiter:
    :return:
    """

    result = pd.read_csv(filepath, sep=delimiter, header=None, names=['nid', 'cid']) \
        .set_index('nid').sort_index()

    if not named:
        # Make sure all indices are 0-indexed
        result.index -= 1
        result['cid'] -= 1

    return result


def write_comlist(comlist: ComList, filepath: str, named=False):
    """
    Writes a community list to a file.

    comlist specification:
    - have two columns, separated by a tab
    - have no header
    - have no comments
    - all indices should be integers
    - all indices should be 1-indexed

    if named is True, then numeric index contraints are ignored

    :param comlist:
    :param filepath:
    :return:
    """
    result = comlist.sort_index().reset_index()

    if not named:
        # Make sure all indices are 1-indexed
        result['nid'] += 1
        result['cid'] += 1

    result.to_csv(filepath, sep='\t', index=False, header=False, columns=['nid', 'cid'])


def read_coms(filepath: str) -> Coms:
    """
    Reads a communities from a file.

    coms specification:
    - have no header
    - have one community per line
    - each line contains a tab-separated list of node indexes
    - all indices should be integers
    - all indices should be 1-indexed

    :param filepath:
    :return:
    """
    with open(filepath, 'r') as f:
        return {
            i: [int(x) - 1 for x in re.split(r'\s|\t', line.strip())]
            for i, line in enumerate(f.readlines())
        }


def write_coms(coms: Coms, filepath: str):
    """
    Writes a communities to a file.

    coms specification:
    - have no header
    - have one community per line
    - each line contains a tab-separated list of node indexes
    - all indices should be integers
    - all indices should be 1-indexed

    :param coms:
    :param filepath:
    :return:
    """
    with open(filepath, 'w') as f:
        for _, coms in sorted(coms.items()):
            f.write(' '.join(map(lambda x: x + 1, coms)))


def coms_to_comlist(coms: Coms) -> ComList:
    return pd.DataFrame([
        (nid, cid)
        for cid, nids in coms.items()
        for nid in nids
    ], columns=['nid', 'cid']).set_index('nid').sort_index()


def comlist_to_coms(comlist: ComList) -> Coms:
    grouped_coms = comlist.reset_index().groupby('cid').apply(lambda x: x.nid.tolist())
    return dict(grouped_coms.iteritems())


def read_edgelist(filepath: str, delimiter='\t') -> EdgeList:
    """
    Reads an edgelist from a file.

    edgelist specification:
    - have two columns, separated by a tab
    - have no header
    - have no comments
    - all indices should be integers
    - all indices should be 1-indexed

    :param filepath: path to the edgelist file
    :param delimiter:
    :return:
    """
    result = pd.read_csv(filepath, sep=delimiter, header=None, names=['src', 'dst']) \
        .sort_values(by=['src', 'dst'], ignore_index=True)

    # Make sure all indices are 0-indexed
    result['src'] -= 1
    result['dst'] -= 1

    return result


def write_edgelist(edges: EdgeList, filepath: str, delimiter='\t'):
    """
    Writes an edgelist to a file.

    edgelist specification:
    - have two columns, separated by a tab
    - have no header
    - have no comments
    - all indices should be integers
    - all indices should be 1-indexed

    :param edges:
    :param filepath:
    :param delimiter:
    :return:
    """
    # Make sure all indices are 1-indexed
    df = edges.sort_values(by=['src', 'dst'], ignore_index=True)
    df['src'] += 1
    df['dst'] += 1
    df.to_csv(filepath, sep=delimiter, index=False, header=False, columns=['src', 'dst'])


def read_edgelist_graph(filepath: str, directed=False) -> ig.Graph:
    return ig.Graph.DataFrame(edges=read_edgelist(filepath), directed=directed)
