import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import numpy as np

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.metrics import nmi, nf1, modularity, conductance
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
        self.LOG = get_logger(self.__class__.__name__)
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
        if 'dynamic' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating dynamic benchmark')
            scores = []
            for i, ground_truth_file in enumerate(
                    sorted(self.context.version.train_part().snapshots.glob('*.comlist'))):
                LOG.debug(f'Evaluating snapshot {i}')
                prediction_file = self.context.output_dir.joinpath(ground_truth_file.name)
                ground_truth = CommunityAssignment.load_comlist(str(ground_truth_file))
                prediction = CommunityAssignment.load_comlist(str(prediction_file))
                score = self.evaluate_single(prediction, ground_truth)
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
            ground_truth_file = self.context.version.train_part().static_ground_truth
            prediction_file = self.context.output_dir.joinpath(ground_truth_file.name)
            ground_truth = CommunityAssignment.load_comlist(str(ground_truth_file))
            prediction = CommunityAssignment.load_comlist(str(prediction_file))
            score = self.evaluate_single(prediction, ground_truth)

            return {
                f'avg_{self.metric_name()}': score
            }

    @abstractmethod
    def evaluate_single(
            self,
            prediction: CommunityAssignment,
            ground_truth: CommunityAssignment,
    ) -> float:
        raise NotImplementedError()


class QualityMetric(EvaluationMetric):
    def evaluate(self) -> dict:
        if 'dynamic' in self.context.benchmark.tags:
            self.LOG.debug('Evaluating dynamic benchmark')
            scores = []
            for i, graph_file in enumerate(
                    sorted(self.context.version.train_part().snapshots.glob('*.edgelist'))):
                LOG.debug(f'Evaluating snapshot {i}')
                prediction_file = self.context.output_dir.joinpath(graph_file.name).with_suffix('.comlist')
                prediction = CommunityAssignment.load_comlist(str(prediction_file))
                graph = read_edgelist_graph(str(graph_file))
                score = self.evaluate_single(graph, prediction)
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
            prediction_file = self.context.output_dir.joinpath(
                self.context.version.train_part().static_ground_truth.name
            )
            prediction = CommunityAssignment.load_comlist(str(prediction_file))
            graph = read_edgelist_graph(
                str(self.context.version.train_part().static),
                directed=self.context.version.get_param('directed', False),
            )
            score = self.evaluate_single(graph, prediction)
            return {
                f'avg_{self.metric_name()}': score
            }

    @abstractmethod
    def evaluate_single(
            self,
            graph: ig.Graph,
            prediction: CommunityAssignment,
    ) -> float:
        raise NotImplementedError()


class MetricNMI(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return nmi(prediction.to_comlist(), ground_truth.to_comlist())

    def metric_name(self) -> str:
        return 'nmi'


class MetricNF1(AnnotatedEvaluationMetric):
    def evaluate_single(self, prediction: CommunityAssignment, ground_truth: CommunityAssignment) -> float:
        return nf1(prediction.to_comms(), ground_truth.to_comms())

    def metric_name(self) -> str:
        return 'nf1'

class MetricModularity(QualityMetric):
    def evaluate_single(
            self,
            graph: ig.Graph,
            prediction: CommunityAssignment,
    ) -> float:
        return modularity(graph, prediction.to_comlist())

    def metric_name(self) -> str:
        return 'modularity'

class MetricConductance(QualityMetric):
    def evaluate_single(
            self,
            graph: ig.Graph,
            prediction: CommunityAssignment,
    ) -> float:
        return conductance(graph, prediction.to_comms())

    def metric_name(self) -> str:
        return 'conductance'


ANNOTATED_METRICS = [
    MetricNMI,
    MetricNF1,
]

QUALITY_METRICS = [
    MetricModularity,
    MetricConductance,
]