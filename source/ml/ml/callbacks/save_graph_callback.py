from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import torch
import wandb
from pytorch_lightning import Callback, Trainer, LightningModule
from sklearn.cluster import DBSCAN
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from datasets.utils.conversion import igraph_from_hetero
from ml.algo.clustering import KMeans
from ml.utils import Metric, HParams
from ml.utils.graph import extract_attribute
from ml.utils.labelling import NodeLabelling
from ml.utils.outputs import OutputExtractor
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class SaveGraphCallbackParams(HParams):
    metric: Metric = Metric.DOTP
    """Metric to use for kmeans clustering."""


class SaveGraphCallback(Callback):
    def __init__(
            self,
            data: HeteroData,
            node_labels: Dict[str, NodeLabelling] = None,
            hparams: SaveGraphCallbackParams = None,
            clustering: bool = False,
    ) -> None:
        super().__init__()
        self.data = data
        self.node_labels = node_labels or {}
        self.hparams = hparams or SaveGraphCallbackParams()
        self.clustering = clustering

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        outputs = OutputExtractor(outputs)

        if self.clustering:
            Z = outputs.extract_cat('X')
        else:
            Z = outputs.extract_cat_kv('Z_dict')

        logger.info("Saving graph")
        allowed_attrs = ['name', 'timestamp_from', 'timestamp_to', 'train_mask', 'test_mask', 'val_mask']

        node_attrs = {}
        edge_attrs = {}
        for attr in allowed_attrs:
            if attr in self.data.keys:
                node_data = extract_attribute(self.data, attr, edge_attr=False, error=False)
                if node_data:
                    node_attrs[attr] = node_data

                edge_data = extract_attribute(self.data, attr, edge_attr=True, error=False)
                if edge_data:
                    edge_attrs[attr] = edge_data

        if 'name' in node_attrs:
            node_attrs['label'] = node_attrs['name']

        node_attrs.update(self.node_labels)
        G, _, _, _ = igraph_from_hetero(self.data, node_attrs=node_attrs, edge_attrs=edge_attrs)

        logger.info('Running K-means before saving graph')
        k = len(torch.unique(torch.cat(list(self.node_labels['Louvain Labels'].values()), dim=0))) \
            if 'Louvain Labels' in self.node_labels else 7
        k = min(k, 24)
        I = KMeans(-1, k, metric=self.hparams.metric).fit(Z).assign(Z)
        G.vs['precluster_km'] = I.numpy()

        if self.clustering and 'z' in outputs:
            logger.info('Saving resulting clustering')
            z = outputs.extract_cat('z')
            G.vs['mgtcom'] = z.numpy()


        save_dir = Path(wandb.run.dir) / 'graph.graphml'
        logger.info(f"Saving graph to {save_dir}")
        G.write_graphml(str(save_dir))
