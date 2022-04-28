from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.embedding_eval_callback import EmbeddingEvalCallback
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallback
from ml.callbacks.lp_eval_callback import LPEvalCallback
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.callbacks.save_graph_callback import SaveGraphCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.layers.embedding import HeteroNodeEmbedding
from ml.models.het2vec import Het2VecModel, Het2VecDataModule, Het2VecDataModuleParams
from ml.models.mgcom_feat import MGCOMTopoDataModule
from ml.utils import dataset_choices, Metric, OptimizerParams


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    metric: Metric = Metric.DOTP
    repr_dim: int = 128
    optimizer_params = OptimizerParams()
    data_params: Het2VecDataModuleParams = Het2VecDataModuleParams()


class Het2VecExecutor(BaseExecutor):
    args: Args
    datamodule: MGCOMTopoDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return Het2VecDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model(self):
        embedder = HeteroNodeEmbedding(
            self.datamodule.data.num_nodes_dict,
            self.args.repr_dim,
        )

        return Het2VecModel(
            embedder=embedder,
            metric=self.args.metric,
            optimizer_params=self.args.optimizer_params,
        )

    def callbacks(self) -> List[Callback]:
        return self._embedding_task_callbacks()

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    Het2VecExecutor().cli()