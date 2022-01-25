import logging
import pathlib
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type, Optional

import cdlib.evaluation as cdlib_eval
import igraph as ig
import numpy as np
import yaml

from benchmarks.config import BenchmarkConfig
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
    cache: dict = field(default_factory=dict)


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

    def load_ground_truth(self, file: Path) -> CommunityAssignment:
        if str(file) in self.context.cache:
            LOG.debug('Loading ground truth from cache')
            ground_truth = self.context.cache[str(file)]
        else:
            ground_truth = CommunityAssignment.load_comlist(str(file))
            self.context.cache[str(file)] = ground_truth

        return ground_truth

    def load_prediction(self, file: Path, info_file: Path) -> Optional[CommunityAssignment]:
        if str(file) in self.context.cache:
            LOG.debug('Loading prediction from cache')
            prediction = self.context.cache[str(file)]
            if prediction.is_empty() and not self.allow_empty_prediction():
                LOG.warning('No communities are found in the prediction')
                return None

            prediction = self.context.cache[str(file) + "_proc"]
        else:
            prediction = CommunityAssignment.load_comlist(str(file))
            self.context.cache[str(file)] = prediction.clone()

            if prediction.is_empty() and not self.allow_empty_prediction():
                LOG.warning('No communities are found in the prediction')
                return np.NAN

            # Add missing nodes to community list
            if info_file.exists():
                LOG.debug('Loading missing nodes if applicable')
                graph_info = yaml.safe_load(info_file.read_text())
                prediction.add_missing_nodes(graph_info['nodes'])

            self.context.cache[str(file) + "_proc"] = prediction

        return prediction

    def load_graph(self, file: Path) -> ig.Graph:
        if str(file) in self.context.cache:
            LOG.debug('Loading graph from cache')
            graph = self.context.cache[str(file)]
        else:
            graph = read_edgelist_graph(
                str(file),
                directed=self.context.version.get_param('directed', False),
            )

            self.context.cache[str(file)] = graph
        return graph


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
        prediction = self.load_prediction(prediction_file, ground_truth_file.with_suffix('.info.yaml'))
        if prediction is None:
            return np.NAN

        ground_truth = self.load_ground_truth(ground_truth_file)

        LOG.debug(f'Evaluating the metric')
        return self.evaluate_single(prediction, ground_truth)

    @abstractmethod
    def evaluate_single(
            self,
            prediction: CommunityAssignment,
            ground_truth: CommunityAssignment,
    ) -> float:
        raise NotImplementedError()

    def allow_empty_prediction(self):
        return False


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
        prediction = self.load_prediction(prediction_file, graph_file.with_suffix('.info.yaml'))
        if prediction is None:
            return np.NAN

        graph = self.load_graph(graph_file)

        LOG.debug(f'Evaluating the metric')
        return self.evaluate_single(graph, prediction)

    @abstractmethod
    def evaluate_single(
            self,
            graph: ig.Graph,
            prediction: CommunityAssignment,
    ) -> float:
        raise NotImplementedError()

    def allow_empty_prediction(self):
        return False


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


class MetricAverageODF(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return cdlib_eval.avg_odf(
            graph,
            prediction.to_nodeclustering(),
        ).score

    def metric_name(self) -> str:
        return 'avg_odf'


class MetricCommunityCount(QualityMetric):
    def evaluate_single(self, graph: ig.Graph, prediction: CommunityAssignment) -> float:
        return prediction.community_count()

    def metric_name(self) -> str:
        return 'community_count'

    def allow_empty_prediction(self):
        return True


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
    metrics.append(MetricNormalizedCut)
    metrics.append(MetricAverageODF)
    metrics.append(MetricModularityOverlap)
    metrics.append(MetricLinkModularity)
    metrics.append(MetricZModularity)
    if overlapping:
        pass
    else:
        metrics.append(MetricModularity)

    return metrics
