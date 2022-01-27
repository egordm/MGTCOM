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

    @abstractmethod
    def metric_name(self) -> str:
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

    @abstractmethod
    def evaluate_step(self) -> float:
        raise NotImplementedError()


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

    def metric_name(self) -> str:
        return 'nmi'


class OverlappingNMI(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.overlapping_normalized_mutual_information_MGH(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'overlapping_nmi'


class MetricNF1(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.nf1(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'nf1'


class MetricOmega(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.omega(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'omega'


class MetricF1(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.f1(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'f1'


class MetricAdjustedRandIndex(AnnotatedEvaluationMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.adjusted_rand_index(
            self.prediction.to_clustering(),
            self.ground_truth.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'adjusted_rand_inde'


class MetricModularity(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.newman_girvan_modularity(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'modularity'


class MetricLinkModularity(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.link_modularity(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'link_modularity'


class MetricModularityOverlap(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.modularity_overlap(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'modularity_overlap'


class MetricZModularity(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.z_modularity(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'z_modularity'


class MetricConductance(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.conductance(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'conductance'


class MetricExpansion(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.expansion(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'expansion'


class MetricInternalDensity(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.internal_edge_density(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'internal_edge_density'


class MetricNormalizedCut(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.normalized_cut(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'normalized_cut'


class MetricAverageODF(QualityMetric):
    def evaluate_step(self) -> float:
        return cdlib_eval.avg_odf(
            self.graph,
            self.prediction.to_clustering(),
        ).score

    def metric_name(self) -> str:
        return 'avg_odf'


class MetricCommunityCount(QualityMetric):
    def evaluate_step(self) -> float:
        return self.prediction.community_count()

    def metric_name(self) -> str:
        return 'community_count'


def get_metric_list(ground_truth: bool, overlapping: bool) -> List[Type[EvaluationMetric]]:
    metrics = []

    metrics.append(MetricConductance)
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
