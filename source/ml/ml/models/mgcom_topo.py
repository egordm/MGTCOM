from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from ml.data.loaders.nodes_loader import NodesLoader, HeteroNodesLoader
from ml.data.samplers.hgt_sampler import HGTSamplerParams, HGTSampler
from ml.data.samplers.hybrid_sampler import HybridSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.transforms.eval_split import EvalNodeSplitTransform
from ml.layers.fc_net import FCNet, FCNetParams
from ml.layers.hybrid_conv_net import HybridConvNet, HybridConvNetParams
from ml.models.node2vec import Node2VecModel
from ml.utils import HParams, DataLoaderParams, Metric, OptimizerParams, OutputExtractor
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class MGCOMTopoModelParams(HParams):
    embed_node_types: List[NodeType] = field(default_factory=list)
    metric: Metric = Metric.L2

    feat_dim: int = 32
    conv_hidden_dim: Optional[int] = None
    conv_num_layers: int = 2
    conv_num_heads: int = 2

    repr_dim: int = 32
    hidden_dim: List[int] = field(default_factory=lambda: [32])


class MGCOMTopoModel(pl.LightningModule):
    hparams: Union[MGCOMTopoModelParams, OptimizerParams]
    val_Z_dict: Dict[NodeType, Tensor] = None
    val_Z: Tensor = None

    def __init__(
            self,
            metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
            hparams: MGCOMTopoModelParams,
            optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.save_hyperparameters(optimizer_params.to_dict())

        self.embedder = HybridConvNet(
            metadata,
            embed_num_nodes={
                node_type: num_nodes
                for node_type, num_nodes in num_nodes_dict.items() if node_type in self.hparams.embed_node_types
            },
            hparams=HybridConvNetParams(
                repr_dim=self.hparams.feat_dim,
                hidden_dim=self.hparams.conv_hidden_dim,
                num_layers=self.hparams.conv_num_layers,
                num_heads=self.hparams.conv_num_heads,
            )
        )
        self.out_net = FCNet(
            self.hparams.feat_dim,
            hparams=FCNetParams(
                repr_dim=self.hparams.repr_dim,
                hidden_dim=self.hparams.hidden_dim,
            )
        )

        # noinspection PyTypeChecker
        self.n2v_model = Node2VecModel(
            embedder=None,
            metric=Metric.L2,
        )

    def forward(self, batch):
        node_meta = batch
        return self.embedder(node_meta)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pos_walks, neg_walks, node_meta = batch

        Z_dict = self.embedder(node_meta)

        # Transform hetero data to homogenous data in the sampled order
        Z = torch.zeros(node_meta.batch_size, self.embedder.repr_dim)
        for store in node_meta.node_stores:
            node_type = store._key
            Z[store.batch_perm] = Z_dict[node_type]

        loss = self.n2v_model.loss(pos_walks, neg_walks, Z)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        epoch_loss = outputs.extract_mean('loss')
        self.log('epoch_loss', epoch_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=self.forward(batch)
        )

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.val_Z_dict = outputs.extract_cat_dict('Z_dict')
        self.val_Z = torch.cat(list(self.val_Z_dict.values()), dim=0)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=self.forward(batch)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


@dataclass
class MGCOMTopoDataModuleParams(HParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()
    hgt_params: HGTSamplerParams = HGTSamplerParams()


class MGCOMTopoDataModule(pl.LightningDataModule):
    hparams: Union[MGCOMTopoDataModuleParams, DataLoaderParams]
    loader_params: DataLoaderParams

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: MGCOMTopoDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.save_hyperparameters(loader_params.to_dict())
        self.loader_params = loader_params

        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = EvalNodeSplitTransform()(dataset.data)

    @property
    def metadata(self) -> Metadata:
        return self.dataset.metadata

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        return self.train_data.num_nodes_dict

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        hdata = self.train_data.to_homogeneous(node_attrs=[], edge_attrs=[], add_node_type=False, add_edge_type=False)
        n2v_sampler = Node2VecSampler(hdata.edge_index, hdata.num_nodes, hparams=self.hparams.n2v_params)
        hgt_sampler = HGTSampler(self.train_data, hparams=self.hparams.hgt_params)
        sampler = HybridSampler(n2v_sampler=n2v_sampler, hgt_sampler=hgt_sampler)

        return NodesLoader(
            self.train_data.num_nodes, transform=sampler,
            shuffle=True,
            **self.loader_params.to_dict(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler = HGTSampler(self.val_data, hparams=self.hparams.hgt_params)

        return HeteroNodesLoader(
            self.val_data.num_nodes_dict, transform=sampler,
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        sampler = HGTSampler(self.test_data, hparams=self.hparams.hgt_params)

        return HeteroNodesLoader(
            self.test_data.num_nodes_dict, transform=sampler,
            shuffle=False,
            **self.loader_params.to_dict(),
        )
