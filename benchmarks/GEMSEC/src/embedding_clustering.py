"""Running the model."""
from pathlib import Path

import networkx as nx

from param_parser import parameter_parser
from print_and_read import graph_reader, exports_path
from model import GEMSECWithRegularization, GEMSEC
from model import DeepWalkWithRegularization, DeepWalk

def load_graph(path):
    path = Path(path).absolute()
    print("Loading graph from {}".format(path))
    train_graph_path = path / "train.gpickle"
    G_train = nx.read_gpickle(train_graph_path)
    return G_train


def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    dataset_path = exports_path / args.dataset / args.dataset_version
    graph = load_graph(dataset_path)

    if args.model == "GEMSECWithRegularization":
        model = GEMSECWithRegularization(args, graph)
    elif args.model == "GEMSEC":
        model = GEMSEC(args, graph)
    elif args.model == "DeepWalkWithRegularization":
        model = DeepWalkWithRegularization(args, graph)
    else:
        model = DeepWalk(args, graph)
    model.train()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
