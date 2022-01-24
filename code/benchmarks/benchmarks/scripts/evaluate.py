import os
from dataclasses import dataclass

from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.evaluation import EvaluationContext, get_metric_list
from shared.cli import parse_args
from shared.constants import BENCHMARKS_RESULTS
from shared.logger import get_logger
from shared.schema import DatasetSchema, TAG_GROUND_TRUTH, TAG_OVERLAPPING
from shared.structs import dict_deep_merge

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True)
    version: str = field(positional=True)
    run_name: str = field(positional=True)


def run(args: Args):
    dataset = DatasetSchema.load_schema(args.dataset)
    version = dataset.get_version(args.version)
    baseline = BenchmarkConfig.load_config(args.baseline)
    LOG.info('Starting evaluation for dataset %s, version %s, baseline %s', args.dataset, args.version, args.baseline)

    context = EvaluationContext(
        dataset,
        version,
        baseline,
        BENCHMARKS_RESULTS.joinpath(baseline.name, args.run_name),
    )

    metrics = get_metric_list(
        TAG_GROUND_TRUTH in dataset.tags,
        TAG_OVERLAPPING in baseline.tags or TAG_OVERLAPPING in dataset.tags,
    )

    result = {}
    for metric_cls in metrics:
        metric = metric_cls(context)
        metric_result = metric.evaluate()
        LOG.info('Evaluation result for metric %s: %s', metric.metric_name(), metric_result)
        result = dict_deep_merge(result, metric_result)

    return result


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
