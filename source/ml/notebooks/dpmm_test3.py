import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Data

from datasets import StarWars
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.algo.dpmm.dpm import DirichletProcessMixture
from ml.algo.dpmm.base import EMCallback
from ml.algo.dpmm.dpm_sc import DirichletProcessMixtureSC, DPMixtureSCParams
from ml.algo.dpmm.statistics import InitMode
from ml.evaluation import community_metrics
from ml.utils.plot import create_colormap, draw_ellipses, plot_scatter


def plot_results(ax, X, z, mus, covs, colors, sup):
    _min, _max = X.min(dim=0).values, X.max(dim=0).values

    if sup:
        draw_ellipses(ax, mus, covs, colors, zorder=1)
    else:
        for i in range(len(mus)):
            draw_ellipses(ax, mus[i], covs[i], colors[i].reshape(1, -1).repeat(2, axis=0), zorder=1)

    plot_scatter(ax, X[:, 0], X[:, 1], facecolors=colors[z], alpha=0.6, zorder=2)

    ax.set_xlim([float(_min[0]), float(_max[0])])
    ax.set_ylim([float(_min[1]), float(_max[1])])
    ax.set_xticks(())
    ax.set_yticks(())


# Number of samples per component
n_samples = 500

# Generate random sample, two components
X_dict = torch.load(
    '/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/StarWars/wandb/offline-run-20220501_232858-3kwsr7fr/files/embeddings_hetero.pt')
# X_dict = torch.load('/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/StarWars/wandb/offline-run-20220430_222415-3ftcu5qf/files/embeddings_hetero.pt')
dataset = StarWars()
# X_dict = torch.load('/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/Cora/wandb/run-20220501_235544-ptk0stcg/files/embeddings_hetero.pt')
# dataset = Cora()

X = torch.cat(list(X_dict.values()), dim=0)


class PlotCallback(EMCallback):
    run: int = -1
    step: int = 0

    def on_after_init_params(self, model: DirichletProcessMixtureSC) -> None:
        super().on_after_init_params(model)
        self.run += 1
        self.step = 0

    def on_after_step(self, model: DirichletProcessMixtureSC, lower_bound: Tensor) -> None:
        fig, (ax_sup, ax_sub) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f'Run {self.run}, step {self.step}')

        plot_results(
            ax_sup,
            X,
            model.predict(X),
            model.mus,
            model.covs,
            create_colormap(model.n_components),
            sup=True,
        )

        plot_results(
            ax_sub,
            X,
            model.predict(X),
            model.mus_sub,
            model.covs_sub,
            create_colormap(model.n_components),
            sup=False,
        )

        fig.show()
        self.step += 1


# Fit a Dirichlet process Gaussian mixture using five components
dpmm = DirichletProcessMixtureSC(DPMixtureSCParams(
    init_k=3,
    init_mode=InitMode.KMEANS,
    prior_alpha=10,
    prior_sigma_scale=0.001,
))
dpmm.fit(X, n_init=1, max_iter=100, callbacks=[PlotCallback()])


def extract_edge_index(data):
    if isinstance(data, HeteroData):
        hdata = to_homogeneous(
            data,
            node_attrs=[], edge_attrs=[],
            add_node_type=False, add_edge_type=False
        )
        return hdata.edge_index
    elif isinstance(data, Data):
        return data.edge_index
    else:
        return None


z = dpmm.predict(X)
edge_index = extract_edge_index(dataset.data)
metrics = community_metrics(z, edge_index)
print(metrics)
print(dpmm.n_components)

plt.show()
