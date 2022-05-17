from dataclasses import dataclass
from typing import Optional, Union, Dict, List

import torch
import torch.nn.functional as F
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.types import STEP_OUTPUT
from simple_parsing import field
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata, NodeType

from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.loaders.chained_loader import ChainedDataLoader
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.cpgnn_sampler import CPGNNSamplerParams, CPGNNSampler
from ml.data.samplers.hgt_sampler import HGTSampler, HGTSamplerParams
from ml.layers.conv.base import HeteroConvLayer
from ml.layers.conv.hgt_cov_net import HGTConvNet
from ml.layers.conv.hybrid_conv_net import HybridConvNet
from ml.layers.conv.sage_conv_net import SAGEConvNet
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.models.het2vec import Het2VecModel
from ml.models.mgcom_feat import MGCOMFeatDataModule
from ml.models.node2vec import Node2VecModelParams
from ml.utils import OptimizerParams, DataLoaderParams, Metric


class CPGNNConvNet(HGTConvNet):
    def convolve(self, data: HeteroData, X_dict: Dict[NodeType, Tensor] = None, k: int = None) -> Dict[
        NodeType, Tensor]:
        # Apply convolutions
        Z_dict = X_dict
        for i in range(self.num_layers if k is None else k + 1):
            conv = self.convs[i]
            Z_dict_new = conv(Z_dict, data.edge_index_dict)

            if self.use_gru:
                Z_dict = {
                    node_type: self.gru_gate(Z_dict_new[node_type], Z_dict[node_type])
                    for node_type in Z_dict.keys()
                }
            else:
                Z_dict = Z_dict_new

        return Z_dict


@dataclass
class CPGNNModelParams(Node2VecModelParams):
    metric: Metric = Metric.L2
    """Metric to use for distance/similarity calculation. (for loss)"""
    k_length: int = field(default=3, cmd=False)
    repr_dim: int = 32
    """Dimension of the representation vectors."""

    num_layers_aux: int = 1
    num_heads: int = 2
    """Hidden dimension of the convolution layers. If None, use repr_dim."""
    use_gru: bool = True


class CPGNNModel(Het2VecModel):
    hparams: Union[CPGNNModelParams, OptimizerParams]

    def __init__(
        self,
        metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
        hparams: CPGNNModelParams,
        optimizer_params: Optional[OptimizerParams] = None,
    ) -> None:
        self.save_hyperparameters(hparams.to_dict())

        embedder = HybridConvNet(
            metadata,
            embed_num_nodes=num_nodes_dict,
            hidden_dim=self.hparams.repr_dim,
            repr_dim=self.hparams.repr_dim,
            use_dropout=False,
            conv=None,
        )
        super().__init__(embedder, hparams, optimizer_params)

        self.aux_conv = SAGEConvNet(
            metadata, self.hparams.repr_dim,
            hidden_dim=self.hparams.repr_dim,
            num_layers=self.hparams.num_layers_aux,
        )

        self.multihead_conv = CPGNNConvNet(
            metadata, self.hparams.repr_dim,
            hidden_dim=self.hparams.repr_dim,
            num_layers=self.hparams.k_length,
            heads=self.hparams.num_heads,
            use_gru=self.hparams.use_gru,
        )

        self.loss_weight = torch.nn.Parameter(torch.zeros(self.hparams.k_length))

        self.sim_fn = self.hparams.metric.pairwise_sim_fn

    def forward(self, batch):
        data, node_perm_dict = batch

        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return self.conv_full(data, batch)[1]
        else:
            return self.embedder(data)

    def conv_full(self, data, data_meta, k: int = None):
        Zp_dict = self.embedder(data, return_raw=True)
        Ca_dict = self.aux_conv(data, Zp_dict, return_raw=True)
        Cp_dict = self.multihead_conv(data, Ca_dict, return_raw=True, k=k)

        Zp_dict = HeteroConvLayer.process_batch(data, Zp_dict)
        Cp_dict = HeteroConvLayer.process_batch(data, Cp_dict)

        return Zp_dict, Cp_dict

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        k, (pos_walks, neg_walks, node_meta) = batch

        _, node_perm_dict = node_meta
        Zp_dict, Cp_dict = self.conv_full(*node_meta, k=k)

        Zp = ToHeteroMappingTransform.inverse_transform_values(
            Zp_dict, node_perm_dict, shape=[self.embedder.repr_dim], device=self.device
        )
        Cp = ToHeteroMappingTransform.inverse_transform_values(
            Cp_dict, node_perm_dict, shape=[self.embedder.repr_dim], device=self.device
        )

        loss = self.loss(k, pos_walks, neg_walks, Zp, Cp)
        return loss

    def _context_score(self, pairs: Tensor, Zp: Tensor, Cp: Tensor) -> Tensor:
        src, dst = pairs[:, 0], pairs[:, 1]

        sim = self.sim_fn(Zp[src] * Cp[src], Zp[dst] * Cp[dst])
        return sim

    def loss(self, k, pos_pairs: Tensor, neg_pairs: Tensor, Zp: Tensor, Cp: Tensor):
        pos_score = self._context_score(pos_pairs, Zp, Cp)
        neg_score = self._context_score(neg_pairs, Zp, Cp)
        pos_loss = -F.logsigmoid(pos_score).view(-1).mean()
        neg_loss = -F.logsigmoid(-neg_score).view(-1).mean()

        # Multi loss
        weight = torch.exp(-self.loss_weight[k])
        return (pos_loss + neg_loss) * weight + self.loss_weight[k]


@dataclass
class CPGNNDataModuleParams(GraphDataModuleParams):
    sampler_params: CPGNNSamplerParams = CPGNNSamplerParams()
    num_layers_aux: int = field(default=2, cmd=False)

    num_samples: List[int] = field(default_factory=lambda: [3, 2, 2, 2])


class CPGNNDataModule(MGCOMFeatDataModule):
    hparams: Union[CPGNNDataModuleParams, DataLoaderParams]

    def _build_n2v_sampler(self, data: HeteroData, transform_meta=None) -> Union[CPGNNSampler]:
        hdata = data.to_homogeneous(
            node_attrs=[], edge_attrs=[],
            add_node_type=False, add_edge_type=False
        )
        sampler = CPGNNSampler(
            hdata.edge_index, hdata.num_nodes,
            hparams=self.hparams.sampler_params, transform_meta=transform_meta,
        )
        return sampler

    def _build_conv_sampler(self, data: HeteroData) -> Union[HGTSampler]:
        assert (self.hparams.sampler_params.k_length) + self.hparams.num_layers_aux == len(self.hparams.num_samples), \
            f"k_length + num_layers_aux != num_samples. {self.hparams.sampler_params.k_length} " \
            f"+ {self.hparams.num_layers_aux} != {len(self.hparams.num_samples)}"

        sampler = HGTSampler(data, hparams=HGTSamplerParams(
            num_samples=self.hparams.num_samples,
        ))
        return sampler

    def train_dataloader(self):
        sampler = self.train_sampler(self.train_data)

        def sampler_fn(k: int):
            def sample(*args, **kwargs):
                return sampler.sample(k, *args, **kwargs)

            return sample

        return ChainedDataLoader([
            NodesLoader(
                self.train_data.num_nodes,
                transform=sampler_fn(k),
                shuffle=True,
                **self.loader_params.to_dict()
            )
            for k in range(self.hparams.sampler_params.k_length)
        ])
