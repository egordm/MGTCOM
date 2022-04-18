from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from datasets.utils.base import Snapshots
from datasets.utils.conversion import igraph_from_hetero
from ml.data.samplers.ballroom_sampler import TemporalNodeIndex, NAN_TIMESTAMP
from ml.data.transforms.ensure_timestamps import EnsureTimestampsTransform
from ml.data.transforms.to_homogeneous import to_homogeneous
from shared import get_logger

logger = get_logger(Path(__file__).stem)


def extract_louvain_labels(data: HeteroData) -> Tensor:
    G, _, _, node_offsets = igraph_from_hetero(data)
    comm = G.community_multilevel()
    return torch.tensor(comm.membership, dtype=torch.long)


# noinspection PyProtectedMember
def extract_timestamp_labels(data: HeteroData) -> Tensor:
    data = EnsureTimestampsTransform(warn=True)(data)

    hdata = to_homogeneous(
        data,
        node_attrs=['timestamp_from'],
        edge_attrs=['timestamp_from'],
        add_node_type=False, add_edge_type=False
    )

    temp_index = TemporalNodeIndex().fit(
        hdata.node_timestamp_from,
        hdata.edge_index,
        hdata.edge_timestamp_from,
    )

    node_timestamps = torch.full([data.num_nodes], -1, dtype=torch.long)
    for (node_id, t) in zip(temp_index.node_ids, temp_index.node_timestamps):
        if node_timestamps[int(node_id)] == -1:
            node_timestamps[int(node_id)] = t

    return node_timestamps


def extract_snapshot_labels(node_timestamps: Tensor, snapshots: Snapshots) -> Tensor:
    snapshots_from = torch.tensor([-1] + [start for (start, end) in snapshots], dtype=torch.long)
    labels = torch.searchsorted(snapshots_from, node_timestamps, side='left') + 1
    return labels
