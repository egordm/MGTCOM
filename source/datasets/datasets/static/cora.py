from pathlib import Path
from typing import Optional, Callable, List

from torch_geometric.data import HeteroData
from torch_geometric.datasets import Planetoid

from datasets.transforms.random_edge_split import RandomEdgeSplit
from datasets.transforms.random_node_split import RandomNodeSplit
from datasets.utils.graph_dataset import DATASET_REGISTRY, BaseGraphDataset
from datasets.utils.labels import extract_louvain_labels
from shared import CACHE_PATH, get_logger

logger = get_logger(Path(__file__).stem)


@DATASET_REGISTRY
class Cora(Planetoid, BaseGraphDataset):
    tags = ['ground-truth']

    def __init__(
            self,
            root: str = None,
            split: str = "public", num_train_per_class: int = 20, num_val: int = 0.1, num_test: int = 0.1,
            transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None
    ):
        if not root:
            root = str(CACHE_PATH.joinpath('dataset', self.name))

        def pre_transform_(data):
            data = data.to_heterogeneous()
            self._extract_labels(data)

            data = RandomNodeSplit(
                split="train_rest",
                num_splits=1,
                num_val=num_val,
                num_test=num_test,
                key=None,
            )(data)

            data = RandomEdgeSplit(
                num_val=num_val,
                num_test=num_test,
                key_prefix='lp_',
                inplace=True,
            )(data)

            return data if pre_transform is None else pre_transform(data)

        super().__init__(root, "Cora", split, num_train_per_class, num_val, num_test, transform, pre_transform_)

    def _extract_labels(self, data: HeteroData):
        logger.info('Extracting Louvain labels')
        louvain = extract_louvain_labels(data)
        for node_type, labels in louvain.items():
            data[node_type].louvain = labels

    def download(self):
        super().download()

    def process(self):
        super().process()

    @staticmethod
    def labels() -> List[str]:
        return ['y', 'louvain']
