from tch_geometric.loader import CustomLoader
from tch_geometric.transforms.constrastive_merge import ContrastiveMergeTransform, EdgeTypeAggregateTransform
from torch_geometric.data import HeteroData

from ml.data.datasets import HeteroEdgesDataset
from ml.data.transforms.unique_nodes import extract_unique_nodes


class ContrastiveTopoDataLoader(CustomLoader):
    def __init__(
            self,
            data: HeteroData,
            neg_sampler,
            neighbor_sampler,
            **kwargs
    ):
        dataset = HeteroEdgesDataset(data, temporal=False)
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

        # Merge pos and neg graphs (stores pos edges and neg edges in the same graph)
        data = self.contrastive_merge(pos_data, neg_data)

        # Combine unique nodes and save perm attribute
        nodes_dict, data = extract_unique_nodes(data)

        # Neighbor samples
        samples = self.neighbor_sampler(nodes_dict)

        # Change node indices to unique indices (perm)
        for store in data.node_stores:
            store.x = store.perm

        # Aggregate pos and neg edges disregarding hetero types
        data_agg = self.edge_type_aggr(data)

        return samples, data_agg, data.node_types
