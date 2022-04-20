from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple

from datasets import GraphDataset
from datasets.transforms.define_snapshots import DefineSnapshots
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams, BallroomSampler
from ml.data.samplers.base import Sampler
from ml.models.node2vec import Node2VecModel, Node2VecDataModule
from ml.utils import DataLoaderParams
from shared import get_logger

EPS = 1e-15

logger = get_logger(Path(__file__).stem)


class Ballroom2VecModel(Node2VecModel):
    pass


@dataclass
class Ballroom2VecDataModuleParams(GraphDataModuleParams):
    window: Optional[Tuple[int, int]] = None
    ballroom_params: BallroomSamplerParams = BallroomSamplerParams()


class Ballroom2VecDataModule(Node2VecDataModule):
    hparams: Union[Ballroom2VecDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: Ballroom2VecDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        if hparams.window is None:
            if isinstance(dataset, GraphDataset) and dataset.snapshots is not None:
                logger.warning("No temporal window specified, trying to infer it from dataset snapshots")
                snapshot_key = max(dataset.snapshots.keys())
                snapshot = dataset.snapshots[snapshot_key]
            else:
                logger.warning('Dataset does not have snapshots, trying to create snapshots from dataset')
                snapshot = DefineSnapshots(10)(dataset.data)

            hparams.window = (0, int(snapshot[0][1] - snapshot[0][0]))
            logger.warning(f"Inferred temporal window: {hparams.window}")

        super().__init__(dataset, hparams, loader_params)

    def train_sampler(self) -> Optional[Sampler]:
        return BallroomSampler(
            self.train_data.node_timestamp_from,
            self.train_data.edge_index,
            self.train_data.edge_timestamp_from,
            tuple(self.hparams.window),
            hparams=self.hparams.ballroom_params
        )

    def eval_sampler(self) -> Optional[Sampler]:
        return None
