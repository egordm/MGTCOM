import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import torch
import wandb
from pytorch_lightning import Callback, Trainer, LightningModule

from datasets import GraphDataset
from datasets.utils.conversion import igraph_from_hetero, extract_attribute_dict
from ml.algo.clustering import KMeans
from ml.models.mgcom_comdet import MGCOMComDetModel
from ml.models.mgcom_e2e import MGCOME2EModel
from ml.utils import Metric, HParams
from ml.utils.outputs import OutputExtractor
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class SaveGraphCallbackParams(HParams):
    metric: Metric = Metric.DOTP
    """Metric to use for kmeans cluster_model."""


class SaveGraphCallback(Callback):
    def __init__(
            self,
            dataset: GraphDataset,
            hparams: SaveGraphCallbackParams = None,
            clustering: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.hparams = hparams or SaveGraphCallbackParams()
        self.clustering = clustering

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        outputs = OutputExtractor(outputs)
        data = self.dataset.data

        if data.num_nodes > 10 or data.num_edges > 100000:
        # if data.num_nodes > 20000 or data.num_edges > 100000:
            logger.info('Graph has too many nodes. Not saving')
            return

        if isinstance(pl_module, MGCOMComDetModel):
            Z = outputs.extract_first('X', device='cpu')
        else:
            Z = outputs.extract_cat_kv('Z_dict', device='cpu')

        logger.info("Saving graph")
        allowed_attrs = ['name', 'timestamp_from', 'timestamp_to', 'train_mask', 'test_mask', 'val_mask']
        allowed_attrs.extend(self.dataset.labels())

        node_attrs = {}
        edge_attrs = {}
        for attr in allowed_attrs:
            if attr in data.keys:
                node_data = extract_attribute_dict(data, attr, edge_attr=False, error=False)
                if node_data:
                    node_attrs[attr] = node_data

                edge_data = extract_attribute_dict(data, attr, edge_attr=True, error=False)
                if edge_data:
                    edge_attrs[attr] = edge_data

        if 'name' in node_attrs:
            node_attrs['label'] = node_attrs['name']

        if 'y' in node_attrs:
            node_attrs['ground_truth_y'] = node_attrs['y']
            del node_attrs['y']

        G, _, _, _ = igraph_from_hetero(data, node_attrs=node_attrs, edge_attrs=edge_attrs)

        logger.info('Running K-means before saving graph')
        k = len(set(G.vs['louvain'])) if 'louvain' in node_attrs else 7 # error
        k = min(k, 24)
        I = KMeans(-1, k, metric=self.hparams.metric).fit(Z).assign(Z)
        G.vs['precluster_km'] = I.numpy() # argument 'input' (position 1) must be Tensor, not dict

        if isinstance(pl_module, MGCOMComDetModel):
            logger.info('Saving resulting cluster_model')
            z = outputs.extract_first('z')
            G.vs['mgtcom'] = z.numpy()
        elif isinstance(pl_module, MGCOME2EModel):
            logger.info('Saving resulting cluster_model')
            if pl_module.cluster_model.is_fitted:
                z = pl_module.cluster_model.predict(Z)
                G.vs['mgtcom'] = z.numpy()
            else:
                logger.info('Cluster model is not fitted. Not saving')

        save_dir = Path(wandb.run.dir) / 'graph.graphml'
        logger.info(f"Saving graph to {save_dir}")
        G.write_graphml(str(save_dir))
