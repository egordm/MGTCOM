import logging
import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Type

import cdlib.evaluation as cdlib_eval
import igraph as ig
import numpy as np

from benchmarks.benchmarks.config import BenchmarkConfig
from shared.graph import CommunityAssignment, read_edgelist_graph
from shared.logger import get_logger
from shared.schema import DatasetSchema, DatasetVersion

LOG = get_logger('Evaluation')


@dataclass
class EvaluationContext:
    dataset: DatasetSchema
    version: DatasetVersion
    benchmark: BenchmarkConfig
    output_dir: Path


class EvaluationMetric:
    LOG: logging.Logger
    context: EvaluationContext

    def __init__(self, context: EvaluationContext) -> None:
        super().__init__()
        self.context = context
        self.LOG = get_logger(self.metric_name())
        self.prepare()

    def prepare(self):
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def metric_name(self) -> str:
        raise NotImplementedError()


class AnnotatedEvaluationMetric(EvaluationMetric):
    def evaluate(self) -> dict:
        PART = self.context.version.train

        if 'dynamic' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating dynamic benchmark')
            scores = []
            for i, ground_truth_file in enumerate(PART.get_snapshot_ground_truths()):
                self.LOG.debug(f'Evaluating snapshot {i}')
                prediction_file = self.context.output_dir.joinpath(ground_truth_file.name)
                score = self._evaluate_by_params(prediction_file, ground_truth_file)
                scores.append(score)
            return {
                f'avg_{self.metric_name()}': np.mean(scores),
                'snapshots': {
                    i: {
                        self.metric_name(): score
                    }
                    for i, score in enumerate(scores)
                }
            }
        elif 'static' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating static benchmark')
            ground_truth_file = PART.static_ground_truth
            prediction_file = self.context.output_dir.joinpath(ground_truth_file.name)
            score = self._evaluate_by_params(prediction_file, ground_truth_file)

            return {
                f'avg_{self.metric_name()}': score
            }

    def _evaluate_by_params(
            self, prediction_file: pathlib.Path, ground_truth_file: pathlib.Path
    ):
        ground_truth = CommunityAssignment.load_comlist(str(ground_truth_file))
        prediction = CommunityAssignment.load_comlist(str(prediction_file))
        if prediction.is_empty():
            LOG.warning('No communities are found in the prediction')
            score = np.NAN
        else:
            score = self.evaluate_single(prediction, ground_truth)

        return score

    @abstractmethod
    def evaluate_single(
            self,
            prediction: CommunityAssignment,
            ground_truth: CommunityAssignment,
    ) -> float:
        raise NotImplementedError()


class QualityMetric(EvaluationMetric):
    def evaluate(self) -> dict:
        PART = self.context.version.train

        if 'dynamic' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating dynamic benchmark')
            scores = []
            for i, graph_file in enumerate(PART.get_snapshot_edgelists()):
                self.LOG.debug(f'Evaluating snapshot {i}')
                prediction_file = self.context.output_dir.joinpath(graph_file.name).with_suffix('.comlist')
                score = self._evaluate_by_params(prediction_file, graph_file)
                scores.append(score)
            return {
                f'{self.metric_name()}_avg': np.mean(scores),
                **{
                    f'{self.metric_name()}_snapshots/snapshot_{i}': score
                    for i, score in enumerate(scores)
                },
            }
        elif 'static' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating static benchmark')
            prediction_file = self.context.output_dir.joinpath(PART.static_ground_truth.name)
            graph_file = PART.static_edgelist
            score = self._evaluate_by_params(prediction_file, graph_file)
            return {
                f'{self.metric_name()}_avg': score
            }

    def _evaluate_by_params(
            self, prediction_file: pathlib.Path, graph_file: pathlib.Path
    ):
        prediction = CommunityAssignment.load_comlist(str(prediction_file))
        graph = read_edgelist_graph(
            str(graph_file),
            directed=self.context.version.get_param('directed', False),
        )
        if prediction.is_empty():
            LOG.warning('No communities are found in the prediction')
            score = np.NAN
        else:
            score = self.evaluate_single(graph, prediction)
        return score

    @abstractmethod
    def evaluate_single(
            self,
            graph: ig.Graph,
            prediction: CommunityAssignment,
    ) -> float:
        raise NotImplementedError()


class MetricNMI(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return cdlib_eval.normalized_mutual_information(
            prediction.to_nodeclustering(),
            ground_truth.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'nmi'


class OverlappingNMI(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return cdlib_eval.overlapping_normalized_mutual_information_MGH(
            prediction.to_nodeclustering(),
            ground_truth.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'overlapping_nmi'


class MetricNF1(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return cdlib_eval.nf1(
            prediction.to_nodeclustering(),
            ground_truth.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'nf1'


class MetricOmega(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return cdlib_eval.omega(
            prediction.to_nodeclustering(),
            ground_truth.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'omega'


class MetricF1(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return cdlib_eval.f1(
            prediction.to_nodeclustering(),
            ground_truth.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'f1'


class MetricAdjustedRandIndex(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return cdlib_eval.adjusted_rand_index(
            prediction.to_nodeclustering(),
            ground_truth.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'adjusted_rand_inde'


class MetricModularity(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.newman_girvan_modularity(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'modularity'


class MetricLinkModularity(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.link_modularity(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'link_modularity'


class MetricModularityOverlap(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.modularity_overlap(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'modularity_overlap'


class MetricZModularity(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.z_modularity(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'z_modularity'


class MetricConductance(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.conductance(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'conductance'


class MetricExpansion(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.expansion(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'expansion'


class MetricInternalDensity(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.internal_edge_density(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'internal_edge_density'


class MetricNormalizedCut(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.normalized_cut(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'normalized_cut'


class MetriAverageODF(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.avg_odf(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'avg_odf'


def get_metric_list(ground_truth: bool, overlapping: bool) -> List[Type[EvaluationMetric]]:
    metrics = []

    if ground_truth:
        metrics.append(MetricNF1)
        metrics.append(OverlappingNMI)
        metrics.append(MetricOmega)
        metrics.append(MetricF1)
        metrics.append(MetricAdjustedRandIndex)

        if overlapping:
            pass
        else:
            metrics.append(MetricNMI)

    # Quality metrics
    metrics.append(MetricConductance)
    metrics.append(MetricExpansion)
    metrics.append(MetricInternalDensity)
    metrics.append(MetricNormalizedCut)
    metrics.append(MetriAverageODF)
    metrics.append(MetricModularityOverlap)
    metrics.append(MetricLinkModularity)
    metrics.append(MetricZModularity)
    if overlapping:
        pass
    else:
        metrics.append(MetricLinkModularity)

    return metrics
