import torch
from jsonargparse import lazy_instance
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from ml import newman_girvan_modularity, igraph_from_hetero
from ml.datasets import DATASET_REGISTRY
from ml.layers.embedding import HGTModule
from ml.layers.initialization import INITIALIZER_REGISTRY
from ml.models.positional import PositionalModel, PositionalDataModule
from shared.constants import BENCHMARKS_RESULTS


class PositionalCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_subclass_arguments(DATASET_REGISTRY.classes, "dataset")
        parser.set_choices("dataset", DATASET_REGISTRY.classes)

        parser.add_subclass_arguments(INITIALIZER_REGISTRY.classes, "initializer")
        parser.set_choices("initializer", INITIALIZER_REGISTRY.classes)

        parser.add_subclass_arguments(PositionalDataModule, "loader")
        parser.add_argument("--n_pre_epochs", type=int, default=8)
        parser.add_argument("--n_comm_epochs", type=int, default=6)

        parser.set_defaults({
            "dataset": {
                "class_path": "ml.datasets.StarWars"
            },
            "loader": {
                "class_path": "ml.models.PositionalDataModule",
                "init_args": {
                    "num_samples": [4, 3],
                    "num_neg_samples": 3,
                    "batch_size": 16,
                }
            },
            # "model.clustering_module": lazy_instance(ExplicitClusteringModule, repr_dim=32),
            "model.clustering_module": {
                "class_path": "ml.layers.ExplicitClusteringModule",
                "init_args": {
                    "repr_dim": 32
                }
            },
            "model.embedding_module": lazy_instance(HGTModule),
            "initializer": {
                "class_path": "ml.layers.LouvainInitializer"
            },
        })

        parser.link_arguments("dataset.data", "initializer.init_args.data", apply_on="instantiate")
        parser.link_arguments("dataset.data", "loader.init_args.data", apply_on="instantiate")
        parser.link_arguments("dataset.metadata", "model.embedding_module.init_args.metadata", apply_on="instantiate")
        parser.link_arguments(
            "initializer.n_clusters", "model.clustering_module.init_args.n_clusters", apply_on="instantiate"
        )


cli = PositionalCli(
    PositionalModel,
    run=False,
    save_config_callback=None,
)
dataset = cli.config_init['dataset']
loader = cli.config_init['loader']
initializer = cli.config_init['initializer']
model: PositionalModel = cli.model

run_name = f'positional-{dataset.name}'
save_dir = BENCHMARKS_RESULTS.joinpath('ml', run_name)
save_dir.mkdir(exist_ok=True, parents=True)

print("Pretraining")
trainer = cli.instantiate_trainer(
    min_epochs=cli.config['n_pre_epochs'],
    max_epochs=cli.config['n_pre_epochs'],
    default_root_dir=str(save_dir),
)
trainer.fit(model, loader)

print("Initialize")
embeddings = model.compute_embeddings(trainer, loader)
centers = initializer.initialize(embeddings)
model.clustering_module.reinit(centers)

print("Cluster-aware training")
model.use_clustering = True
trainer = cli.instantiate_trainer(
    min_epochs=cli.config['n_comm_epochs'],
    max_epochs=cli.config['n_comm_epochs'],
    default_root_dir=str(save_dir),
)
trainer.fit(model, loader)

print('Calculating node assignments')
embeddings = model.compute_embeddings(trainer, loader)
I = {k: model.clustering_module.assign(emb).detach().cpu() for k, emb in embeddings.items()}

m = newman_girvan_modularity(dataset.data, I, model.clustering_module.n_clusters)
print(f'Modularity: {m:.4f}')

print('Saving results')
G, _, _ = igraph_from_hetero(dataset.data, node_attrs=dict(comm=I))
G.write_graphml(str(save_dir.joinpath('graph.graphml')))

state = model.state_dict()
torch.save(state, str(save_dir.joinpath('model.pt')))
