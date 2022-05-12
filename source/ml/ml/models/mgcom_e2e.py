from dataclasses import dataclass
from typing import Dict, Union, Optional, overload, Any

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from torch_geometric.typing import Metadata, NodeType

from ml.algo.dpmm.dpmsc import DPMSCHParams, DPMSC
from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.loaders.nodes_loader import HeteroNodesLoader
from ml.layers.loss.isometric_loss import IsometricLoss
from ml.models.mgcom_combi import MGCOMCombiModel, MGCOMCombiModelParams, MGCOMCombiDataModule
from ml.utils import OptimizerParams
from ml.utils.training import ClusteringStage


@dataclass
class MGCOME2EModelParams(MGCOMCombiModelParams):
    use_cluster: bool = True

    cluster_params: DPMSCHParams = DPMSCHParams()
    cluster_weight: float = 0.1

    n_pretrain_epochs: int = 50
    n_feat_epochs: int = 20
    n_cluster_epochs: int = 100


class MGCOME2EModel(MGCOMCombiModel):
    hparams: Union[MGCOME2EModelParams, OptimizerParams]

    def __init__(
        self,
        metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
        hparams: MGCOME2EModelParams,
        optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__(metadata, num_nodes_dict, hparams, optimizer_params)

        self.cluster_model = DPMSC(hparams.cluster_params)

        self.cluster_loss_fn = IsometricLoss(self.hparams.metric)

        self.stage = ClusteringStage.Feature
        self.r_prev = None
        self.sample_space_version =1

    def forward_homogenous(self, batch):
        _, node_perm_dict = batch
        X_dict = self(batch)
        X = ToHeteroMappingTransform.inverse_transform_values(
            X_dict, node_perm_dict, shape=[self.repr_dim], device=self.device
        )
        return X

    def training_step_cluster(self, batch, Z_topo: Tensor, Z_tempo: Tensor):
        if self.cluster_model.n_components <= 1 or self.r_prev is None:
            return None

        (topo_pos_walks, _, _), (tempo_pos_walks, _, _) = batch
        if self.hparams.use_topo and self.hparams.use_tempo:
            Z_combi = self.embedding_combine_fn([Z_topo, Z_tempo])
            idx = topo_pos_walks[:, 0]
        else:
            Z_combi = Z_topo if self.hparams.use_topo else Z_tempo
            idx = topo_pos_walks[:, 0] if self.hparams.use_topo else tempo_pos_walks[:, 0]

        log_r = self.cluster_model.estimate_log_resp(Z_combi)
        log_rk = log_r[idx, :]
        z_k = self.r_prev[idx, :].argmax(dim=-1)

        mus = self.cluster_model.cluster_params.mus
        loss_cluster = self.cluster_loss_fn(Z_combi[idx, :], self.r_prev[idx, :], mus )


        # m = 8
        # log_r = (log_rk / m) - torch.logsumexp(log_rk / m, dim=-1, keepdim=True)
        # r = torch.exp(log_r)
        #
        # r_mask = torch.zeros_like(r, dtype=torch.bool, device=self.device)
        # r_mask[torch.arange(r.shape[0]), z_k] = 1
        #
        # r_pos, r_neg = r[r_mask][:, None], r[~r_mask].view(r.shape[0], -1)
        # diff = torch.relu(0.5 + r_neg - r_pos)
        # loss_cluster = diff.sum(dim=-1).mean()

        # loss_cluster =torch.kl_div(log_r[idx, :], self.r_prev[idx, :])
        # loss_cluster = torch.nn.KLDivLoss(reduction='batchmean')(log_r[idx, :], self.r_prev[idx, :])

        return loss_cluster
        # return r_pos.sum() * -1

        # return loss_cluster
        return 0

    def training_step(self, batch, batch_idx, r=None) -> STEP_OUTPUT:
        (topo_walks, tempo_walks) = batch

        loss_topo, Z_topo = self.training_step_topo(topo_walks)
        loss_tempo, Z_tempo = self.training_step_tempo(tempo_walks)

        loss, out = 0.0, {}
        if loss_topo is not None:
            loss += self.hparams.topo_weight * loss_topo
            out['loss_topo'] = loss_topo.detach()
        if loss_tempo is not None:
            loss += self.hparams.tempo_weight * loss_tempo
            out['loss_tempo'] = loss_tempo.detach()

        if self.hparams.use_cluster:
            loss_cluster = self.training_step_cluster(batch, Z_topo, Z_tempo)
            if loss_cluster is not None:
                loss += self.hparams.cluster_weight * loss_cluster
                out['loss_cluster'] = loss_cluster.detach()

        out['loss'] = loss
        return out

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        out = super().test_step(batch, batch_idx)

        if self.cluster_model.is_fitted:
            _, node_perm_dict = batch
            X_dict = out['Z_dict']
            X = ToHeteroMappingTransform.inverse_transform_values(
                X_dict, node_perm_dict, shape=[self.repr_dim], device=self.device
            )
            z = self.cluster_model.predict(X)

            out.update({
                'X': X,
                'z': z,
            })

        return out

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        return {
            **super().get_progress_bar_dict(),
            'stage': 'feat' if self.stage == ClusteringStage.Feature else 'clus',
            'k': self.cluster_model.n_components,
        }

    def get_extra_state(self) -> Any:
        return {
            'cluster_params': self.cluster_model._get_params(),
            'cluster_prior': self.cluster_model._get_params_prior(),
        }

    def set_extra_state(self, state: Any):
        self.cluster_model._set_params(state['cluster_params'])
        self.cluster_model._set_params_prior(state['cluster_prior'])
        if self.cluster_model.clusters.params is not None:
            self.cluster_model.is_fitted = True


class MGCOME2EDataModule(MGCOMCombiDataModule):
    def cluster_dataloader(self) -> TRAIN_DATALOADERS:
        return HeteroNodesLoader(
            self.train_data.num_nodes_dict,
            transform_nodes_fn=self.eval_sampler(self.train_data),
            shuffle=False,
            **self.loader_params.to_dict()
        )
