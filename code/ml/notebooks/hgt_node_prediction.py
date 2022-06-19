import os.path as osp
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear

from ml.callbacks.progress_bar import CustomProgressBar
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.hgt_sampler import HGTSamplerParams, HGTSampler
from ml.layers.conv.hgt_cov_net import HGTConvNet
from ml.layers.conv.hybrid_conv_net import HybridConvNet
from ml.utils import OutputExtractor

path = osp.join(osp.dirname(osp.realpath(__file__)), './data/DBLP')
dataset = DBLP(path)
data = dataset[0]
print(data)

hgt_sampler = HGTSampler(data, hparams=HGTSamplerParams(num_samples=[6, 12]))


def transform_fn(node_ids: Tensor):
    return hgt_sampler({
        'author': node_ids,
    })


loader = NodesLoader(
    data['author'].num_nodes,
    transform=transform_fn,
    batch_size=100, num_workers=0
)
test = next(iter(loader))
u = 0


class HGTNodePrediction(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.conv = HybridConvNet(
            metadata=data.metadata(),
            embed_num_nodes={
                'conference': data['conference'].num_nodes,
                'author': data['author'].num_nodes,
            },
            conv=HGTConvNet(
                metadata=data.metadata(),
                repr_dim=32
            )
        )
        self.lin = Linear(32, 4)

    def forward(self, data: HeteroData):
        return self.conv(data)

    def training_step(self, batch) -> STEP_OUTPUT:
        data = batch

        Z_dict = self.forward(data)
        Z = self.lin(Z_dict['author'])
        y = data['author'].y[:data['author'].batch_size]

        return {
            'loss': F.cross_entropy(Z, y),
            'hits': Z.argmax(-1) == y
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        hits = outputs.extract_cat('hits')
        acc = hits.sum().item() / hits.numel()
        self.log('train_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)


model = HGTNodePrediction()

trainer = Trainer(max_epochs=200, callbacks=[CustomProgressBar()])
trainer.fit(model, loader)
