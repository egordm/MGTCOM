from pathlib import Path
from typing import Dict, List, Any

import torch
import wandb
from pytorch_lightning import Callback, Trainer, LightningModule
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from datasets.utils.conversion import igraph_from_hetero
from ml.algo.clustering import KMeans
from ml.utils import OutputExtractor
from ml.utils.graph import extract_attribute
from ml.utils.labelling import NodeLabelling
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class SaveGraphCallback(Callback):
    def __init__(self, data: HeteroData, node_labels: Dict[str, NodeLabelling] = None) -> None:
        super().__init__()
        self.data = data
        self.node_labels = node_labels or {}

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        outputs = OutputExtractor(outputs)
        Z_dict = outputs.extract_cat_dict('Z_dict')
        Z = torch.cat(list(Z_dict.values()), dim=0)

        logger.info('Running K-means before saving graph')
        k = len(torch.unique(torch.cat(list(self.node_labels['Louvain Labels'].values()), dim=0))) \
            if 'Louvain Labels' in self.node_labels else 7
        I = KMeans(-1, k).fit(Z).assign(Z)

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
        G.vs['precluster_km'] = I.numpy()

        save_dir = Path(wandb.run.dir) / 'graph.graphml'
        logger.info(f"Saving graph to {save_dir}")
        G.write_graphml(str(save_dir))
