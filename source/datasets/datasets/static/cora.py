from typing import Optional, Callable, List

from torch_geometric.datasets import Planetoid

from datasets.transforms.random_node_split import RandomNodeSplit
from datasets.utils.base import DATASET_REGISTRY, GraphDataset, BaseGraphDataset
from shared import CACHE_PATH


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

            data = RandomNodeSplit(
                split="train_rest",
                num_splits=1,
                num_val=num_val,
                num_test=num_test,
                key=None,
            )(data)

            return data if pre_transform is None else pre_transform(data)

        super().__init__(root, "Cora", split, num_train_per_class, num_val, num_test, transform, pre_transform_)
        u = 0

    def download(self):
        super().download()

    def process(self):
        super().process()

    @staticmethod
    def labels() -> List[str]:
        return ['y']


