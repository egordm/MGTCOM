import torch
from tch_geometric.loader import CustomLoader
from tch_geometric.transforms.constrastive_merge import EdgeTypeAggregateTransform, ContrastiveMergeTransform
from torch.utils.data import Dataset
from tch_geometric.transforms.hgt_sampling import NAN_TIMESTAMP

from ml import extract_unique_nodes


class ContrastiveDataLoader(CustomLoader):
    def __init__(
            self,
            dataset: Dataset,
            neg_sampler,
            neighbor_sampler,
            **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.neg_sampler = neg_sampler
        self.neighbor_sampler = neighbor_sampler
        self.contrastive_merge = ContrastiveMergeTransform()
        self.edge_type_aggr = EdgeTypeAggregateTransform()

    def sample(self, inputs):
        pos_data = inputs

        # Extract central `query` nodes
        ctr_nodes = {
            store._key: store.x[:store.sample_count]
            for store in pos_data.node_stores
        }

        # Sample negative nodes
        neg_data = self.neg_sampler(ctr_nodes)

        # Merge pos and neg graphs
        data = self.contrastive_merge(pos_data, neg_data)

        # Combine unique nodes
        nodes_dict, data = extract_unique_nodes(data)

        # Neighbor samples
        nodes_timestamps_dict = { # TODO: take timestamps from main data object
            node_type: torch.full((len(idx),), NAN_TIMESTAMP, dtype=torch.int64)
            for node_type, idx in nodes_dict.items()
        }

        samples = self.neighbor_sampler(nodes_dict, nodes_timestamps_dict)

        # Aggregate pos and neg edges
        for store in data.node_stores:
            store.x = store.perm
        data_agg = self.edge_type_aggr(data)

        return samples, data_agg, data.node_types
