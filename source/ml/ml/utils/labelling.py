from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from datasets.utils.base import Snapshots
from datasets.utils.conversion import igraph_from_hetero
from ml.data.samplers.ballroom_sampler import TemporalNodeIndex, NAN_TIMESTAMP
from ml.data.transforms.ensure_timestamps import EnsureTimestampsTransform
from ml.data.transforms.to_homogeneous import to_homogeneous
from shared import get_logger

logger = get_logger(Path(__file__).stem)

NodeLabelling = Dict[NodeType, Tensor]


def extract_louvain_labels(data: HeteroData) -> NodeLabelling:
    G, _, _, node_offsets = igraph_from_hetero(data)
    comm = G.community_multilevel()
    membership = torch.tensor(comm.membership, dtype=torch.long)
    membership_dict = {
        node_type: membership[node_offsets[node_type]:node_offsets[node_type] + num_nodes]
        for node_type, num_nodes in data.num_nodes_dict.items()
    }
    return membership_dict


# noinspection PyProtectedMember
def extract_timestamp_labels(data: HeteroData) -> NodeLabelling:
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

    node_offsets = {}
    counter = 0
    for node_type, num_nodes in data.num_nodes_dict.items():
        node_offsets[node_type] = counter
        counter += num_nodes

    node_timestamps_dict = {
        node_type: node_timestamps[node_offsets[node_type]:node_offsets[node_type] + num_nodes]
        for node_type, num_nodes in data.num_nodes_dict.items()
    }

    return node_timestamps_dict


def extract_snapshot_labels(node_timestamps_dict: Dict[NodeType, Tensor], snapshots: Snapshots) -> NodeLabelling:
    snapshots_from = torch.tensor([-1] + [start for (start, end) in snapshots], dtype=torch.long)
    labels = {
        node_type: torch.searchsorted(snapshots_from, node_timestamps, side='left') + 1
        for node_type, node_timestamps in node_timestamps_dict.items()
    }
    return labels
