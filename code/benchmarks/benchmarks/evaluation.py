import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type

import cdlib.evaluation as cdlib_eval
import igraph as ig
import numpy as np
import wandb

from benchmarks.config import BenchmarkConfig
from shared.exceptions import NoCommunitiesFoundError
from shared.graph import CommunityAssignment, read_edgelist_graph
from shared.logger import get_logger
from shared.schema import DatasetSchema, DatasetVersion
from shared.structs import CacheDict

LOG = get_logger('Evaluation')


@dataclass
class EvaluationContext:
    dataset: DatasetSchema
    version: DatasetVersion
    benchmark: BenchmarkConfig
    output_dir: Path
    cache: CacheDict = field(default_factory=CacheDict)


class EvaluationMetric:
    LOG: logging.Logger
    context: EvaluationContext

    def __init__(self, context: EvaluationContext) -> None:
        super().__init__()
        self.context = context
        self.LOG = get_logger(self.metric_name())

    def prepare(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def metric_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def metric_order(cls) -> str:
        raise NotImplementedError()


class QualityMetric(EvaluationMetric):
    graph: ig.Graph = None
    prediction: CommunityAssignment = None

    def load_graph(self, path: Path) -> ig.Graph:
        graph = read_edgelist_graph(
            str(path),
            directed=self.context.version.get_param('directed', False),
        )
        graph.vs['gid'] = graph.vs['name']
        return graph

    def load_comlist(self, path: Path):
        comms = CommunityAssignment.load_comlist(str(path))
        if comms.is_empty():
            raise NoCommunitiesFoundError('No communities found in file {}'.format(path))

        if not self.graph:
            raise ValueError('Graph has not been loaded yet')

        comms = comms.with_nodes(self.graph.vs['gid'])
        return comms

    def prepare(self, graph_path: Path, prediction_path: Path, **kwargs):
        self.graph = self.context.cache.get(str(graph_path), lambda: self.load_graph(graph_path))
        self.prediction = self.context.cache.get(str(prediction_path), lambda: self.load_comlist(prediction_path))

    def evaluate(self) -> dict:
        PART = self.context.version.train

        scores = []
        if 'dynamic' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating dynamic benchmark')
            for i, graph_file in enumerate(PART.get_snapshot_edgelists()):
                self.LOG.debug(f'Evaluating snapshot {i}')
                self.prepare(
                    graph_path=graph_file,
                    prediction_path=self.context.output_dir.joinpath(graph_file.name).with_suffix('.comlist'),
                )
                scores.append(self.evaluate_step())
        elif 'static' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating static benchmark')
            self.prepare(
                graph_path=PART.static_edgelist,
                prediction_path=self.context.output_dir.joinpath(PART.static_ground_truth.name),
            )
            scores.append(self.evaluate_step())

        return {
            self.metric_name(): np.mean(scores),
            f'snapshots/{self.metric_name()}': scores,
        }

    def evaluate_step(self) -> float:
        self.calculate(self.graph, self.prediction)

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        pass


class AnnotatedEvaluationMetric(QualityMetric, ABC):
    ground_truth: CommunityAssignment = None

    def prepare(self, graph_path: Path, **kwargs):
        super().prepare(graph_path, **kwargs)
        ground_truth_path = graph_path.with_suffix('.comlist')
        self.ground_truth = self.context.cache.get(str(ground_truth_path), lambda: self.load_comlist(ground_truth_path))


class MetricNMI(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.normalized_mutual_information(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    @classmethod
    def metric_name(cls) -> str:
        return 'nmi'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'


class OverlappingNMI(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.overlapping_normalized_mutual_information_MGH(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    @classmethod
    def metric_name(cls) -> str:
        return 'overlapping_nmi'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'


class MetricNF1(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.nf1(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    @classmethod
    def metric_name(cls) -> str:
        return 'nf1'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'


class MetricOmega(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.omega(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    @classmethod
    def metric_name(cls) -> str:
        return 'omega'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'


class MetricF1(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.f1(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    @classmethod
    def metric_name(cls) -> str:
        return 'f1'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'


class MetricAdjustedRandIndex(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.adjusted_rand_index(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    @classmethod
    def metric_name(cls) -> str:
        return 'adjusted_rand_inde'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'


class MetricModularity(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'modularity'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.newman_girvan_modularity(graph, clustering.to_clustering()).score


class MetricLinkModularity(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'link_modularity'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.link_modularity(graph, clustering.to_clustering()).score


class MetricModularityOverlap(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'modularity_overlap'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.modularity_overlap(graph, clustering.to_clustering()).score


class MetricZModularity(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'z_modularity'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.z_modularity(graph, clustering.to_clustering()).score


class MetricConductance(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'conductance'

    @classmethod
    def metric_order(cls) -> str:
        return 'minimize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.conductance(graph, clustering.to_clustering()).score


class MetricExpansion(QualityMetric):

    @classmethod
    def metric_name(cls) -> str:
        return 'expansion'

    @classmethod
    def metric_order(cls) -> str:
        return 'minimize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.expansion(graph, clustering.to_clustering()).score


class MetricInternalDensity(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'internal_edge_density'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.internal_edge_density(graph, clustering.to_clustering()).score


class MetricNormalizedCut(QualityMetric):
    @classmethod
    def metric_name(cls) -> str:
        return 'normalized_cut'

    @classmethod
    def metric_order(cls) -> str:
        return 'maximize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.normalized_cut(graph, clustering.to_clustering()).score


class MetricAverageODF(QualityMetric):

    @classmethod
    def metric_name(cls) -> str:
        return 'avg_odf'

    @classmethod
    def metric_order(cls) -> str:
        return 'minimize'

    @classmethod
    def calculate(cls, graph: ig.Graph, clustering: CommunityAssignment) -> float:
        return cdlib_eval.avg_odf(graph, clustering.to_clustering()).score


class MetricCommunityCount(QualityMetric):
    def evaluate_step(self) -> float:
        return self.prediction.community_count()

    @classmethod
    def metric_name(cls) -> str:
        return 'community_count'

    @classmethod
    def metric_order(cls) -> str:
        return 'minimize'


def get_metric_list(ground_truth: bool, overlapping: bool) -> List[Type[EvaluationMetric]]:
    metrics = []

    metrics.append(MetricCommunityCount)

    if ground_truth:
        metrics.append(MetricNF1)
        metrics.append(OverlappingNMI)
        metrics.append(MetricOmega)
        metrics.append(MetricF1)

        if overlapping:
            pass
        else:
            metrics.append(MetricAdjustedRandIndex)
            metrics.append(MetricNMI)

    # Quality metrics
    metrics.append(MetricConductance)
    metrics.append(MetricExpansion)
    metrics.append(MetricInternalDensity)
    # metrics.append(MetricNormalizedCut)
    metrics.append(MetricAverageODF)
    metrics.append(MetricModularityOverlap)
    metrics.append(MetricLinkModularity)
    metrics.append(MetricZModularity)
    if overlapping:
        pass
    else:
        metrics.append(MetricModularity)

    return metrics


ALL_METRICS: List[Type[EvaluationMetric]] = [
    MetricNMI,
    MetricNF1,
    OverlappingNMI,
    MetricOmega,
    MetricF1,
    MetricAdjustedRandIndex,
    MetricModularity,
    MetricModularityOverlap,
    MetricConductance,
    MetricExpansion,
    MetricInternalDensity,
    MetricNormalizedCut,
    MetricAverageODF,
    MetricLinkModularity,
    MetricZModularity,
]

