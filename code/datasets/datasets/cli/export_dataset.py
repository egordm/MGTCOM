from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import HeteroData

from datasets import StarWars, GraphDataset
from datasets.transforms.eval_edge_split import EvalEdgeSplitTransform
from datasets.transforms.homogenify import homogenify
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.utils import HParams, dataset_choices
from shared import parse_args, EXPORTS_PATH, get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class Args(HParams):
    dataset: str = dataset_choices()
    version: str = "base"

    split_force: bool = False
    """Whether to force resplit the dataset. If false, predefined splits will be preferred."""
    split_num_val: float = 0.1
    """Fraction of the dataset to use for validation."""
    split_num_test: float = 0.1
    """Fraction of the dataset to use for testing."""
    homogeneous: bool = False


def run():
    args: Args = parse_args(Args)[0]

    dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()
    data = dataset.data

    train_data, val_data, test_data = EvalEdgeSplitTransform(
        force_resplit=args.split_force,
        num_val=args.split_num_val,
        num_test=args.split_num_test,
        key_prefix='lp_'
    )(data)

    if args.homogeneous:
        train_data = homogenify(train_data)
        val_data = homogenify(val_data)
        test_data = homogenify(test_data)

    train_G = data_to_nx(train_data)
    val_G = data_to_nx(val_data)
    test_G = data_to_nx(test_data)

    save_path = EXPORTS_PATH / args.dataset / args.version
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving train graph to {save_path}/*.gpickle")
    nx.write_gpickle(train_G, save_path / "train.gpickle", protocol=2)
    nx.write_gpickle(val_G, save_path / "val.gpickle", protocol=2)
    nx.write_gpickle(test_G, save_path / "test.gpickle", protocol=2)

    torch.save(train_data, save_path / "train.pt")
    torch.save(val_data, save_path / "val.pt")
    torch.save(test_data, save_path / "test.pt")


def data_to_nx(data: HeteroData):
    hdata = data.to_homogeneous(node_attrs=[], edge_attrs=[], add_node_type=False, add_edge_type=False)

    edgelist = torch.unique(hdata.edge_index.t(), dim=0)

    G = nx.Graph()
    G.add_nodes_from(range(hdata.num_nodes))
    G.add_edges_from(edgelist.tolist())

    return G


if __name__ == '__main__':
    run()
