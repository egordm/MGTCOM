from pathlib import Path
from typing import Dict

from pytorch_lightning import Callback, Trainer, LightningModule
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from datasets.utils.conversion import igraph_from_hetero
from ml.utils.graph import extract_attribute
from ml.utils.labelling import NodeLabelling
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class SaveGraphCallback(Callback):
    def __init__(self, data: HeteroData, node_labels: Dict[str, NodeLabelling] = None) -> None:
        super().__init__()
        self.data = data
        self.node_labels = node_labels or {}

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
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

        for train_logger in trainer.loggers:
            if train_logger.save_dir is not None:
                save_dir = Path(train_logger.save_dir) / 'graph.graphml'
                logger.info(f"Saving graph to {save_dir}")
                G.write_graphml(str(save_dir))
