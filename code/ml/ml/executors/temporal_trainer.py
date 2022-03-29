from pathlib import Path

import torch
from jsonargparse import lazy_instance
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from ml import newman_girvan_modularity, igraph_from_hetero
from ml.datasets import DATASET_REGISTRY
from ml.layers import INITIALIZER_REGISTRY
from ml.layers.embedding import HGTModule
from ml.models import TemporalDataModule, TemporalModel
from ml.models.positional import PositionalModel, PositionalDataModule
from shared.constants import BENCHMARKS_RESULTS


class TemporalCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_subclass_arguments(DATASET_REGISTRY.classes, "dataset")
        parser.set_choices("dataset", DATASET_REGISTRY.classes)

        parser.add_subclass_arguments(INITIALIZER_REGISTRY.classes, "initializer")
        parser.set_choices("initializer", INITIALIZER_REGISTRY.classes)

        parser.add_class_arguments(TemporalDataModule, "loader")
        parser.add_argument("--n_pre_epochs", type=int, default=0)
        parser.add_argument("--n_comm_epochs", type=int, default=6)
        parser.add_argument("--load_dir", type=str)

        parser.add_argument("--repr_dim", type=int, default=32)
        parser.add_argument("--temp_dim", type=int, default=8)

        parser.set_defaults({
            "dataset": {
                "class_path": "ml.datasets.StarWars"
            },
            "loader.num_samples": [4, 3],
            "loader.num_neg_samples": 3,
            "loader.repeat_count": 2,
            "loader.batch_size": 16,
            "loader.window": [0, 1],
            "model.clustering_module": {
                "class_path": "ml.layers.ExplicitClusteringModule",
            },
            "model.temp_clustering_module": {
                "class_path": "ml.layers.ExplicitClusteringModule",
            },
            "model.embedding_module": {
                "class_path": "ml.layers.embedding.HGTModule",
            },
            "model.temp_embedding_module": {
                "class_path": "ml.layers.embedding.HGTModule",
            },
            "initializer": {
                "class_path": "ml.layers.initialization.RandomInitializer",
                "init_args": {
                    "k": 5,
                }
            },
        })

        parser.link_arguments("repr_dim", "model.clustering_module.init_args.repr_dim", apply_on="parse")
        parser.link_arguments("repr_dim", "model.embedding_module.init_args.repr_dim", apply_on="parse")
        parser.link_arguments("temp_dim", "model.temp_embedding_module.init_args.repr_dim", apply_on="parse")
        parser.link_arguments("temp_dim", "model.temp_clustering_module.init_args.repr_dim", apply_on="parse")

        parser.link_arguments("dataset.data", "initializer.init_args.data", apply_on="instantiate")
        parser.link_arguments("dataset.data", "loader.data", apply_on="instantiate")
        parser.link_arguments("dataset.metadata", "model.embedding_module.init_args.metadata", apply_on="instantiate")
        parser.link_arguments("dataset.metadata", "model.temp_embedding_module.init_args.metadata", apply_on="instantiate")
        parser.link_arguments("initializer.n_clusters", "model.clustering_module.init_args.n_clusters",
                              apply_on="instantiate")
        parser.link_arguments("initializer.n_clusters", "model.temp_clustering_module.init_args.n_clusters",
                              apply_on="instantiate")

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()

        # Load weights and parameters
        load_dir = Path(self.config['load_dir'])
        load_state = torch.load(load_dir.joinpath('model.pt'))
        loaded_centroids = load_state['clustering_module.centroids.weight']
        n_clusters, repr_dim = loaded_centroids.shape
        assert self.config['repr_dim'] == repr_dim, f'Expected repr_dim={repr_dim} but got {self.config["repr_dim"]}'
        print(f'Loaded n_clusters={n_clusters} clusters of repr_dim={repr_dim} dimensions')
        self.config['model']['clustering_module']['init_args']['n_clusters'] = n_clusters
        self.config['model']['temp_clustering_module']['init_args']['n_clusters'] = n_clusters
        self.config['initializer']['init_args']['k'] = n_clusters
        self.load_state = load_state

    def instantiate_classes(self) -> None:
        super().instantiate_classes()

        # Load model weights
        model = self.config_init['model']
        state = model.state_dict()
        state.update(self.load_state)
        model.load_state_dict(state)

        # Freeze trained weights
        model.embedding_module.requires_grad_(False)
        model.clustering_module.requires_grad_(False)


def train(cli: TemporalCli) -> None:
    dataset = cli.config_init['dataset']
    loader = cli.config_init['loader']
    initializer = cli.config_init['initializer']
    model: PositionalModel = cli.model

    run_name = f'temporal-{dataset.name}'
    root_dir = BENCHMARKS_RESULTS.joinpath('ml', run_name)
    root_dir.mkdir(exist_ok=True, parents=True)

    if cli.config['n_pre_epochs'] > 0:
        print("Pretraining")
        trainer = cli.instantiate_trainer(
            min_epochs=cli.config['n_pre_epochs'], max_epochs=cli.config['n_pre_epochs'],
            default_root_dir=str(root_dir),
        )
        trainer.fit(model, loader)

        print("Initialize")
        embeddings = model.compute_embeddings(trainer, loader)
        centers = initializer.initialize(embeddings)
        model.clustering_module.reinit(centers)

    print("Cluster-aware training")
    model.use_clustering = True
    trainer = cli.instantiate_trainer(
        min_epochs=cli.config['n_comm_epochs'], max_epochs=cli.config['n_comm_epochs'],
        default_root_dir=str(root_dir),
    )
    trainer.fit(model, loader)

    print('Calculating node assignments')
    embeddings = model.compute_embeddings(trainer, loader)
    I = {k: model.compute_assignments(emb).detach().cpu() for k, emb in embeddings.items()}

    m = newman_girvan_modularity(dataset.data, I, model.clustering_module.n_clusters)
    print(f'Modularity: {m:.4f}')

    print('Saving results')
    save_dir = Path(trainer.log_dir)
    G, _, _, _ = igraph_from_hetero(dataset.data, node_attrs=dict(comm=I))
    G.write_graphml(str(save_dir.joinpath('graph.graphml')))

    state = model.state_dict()
    torch.save(state, str(save_dir.joinpath('model.pt')))


if __name__ == '__main__':
    cli = TemporalCli(
        TemporalModel,
        run=False,
        save_config_callback=None,
    )
    train(cli)
