import os
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.evaluation import EvaluationContext, ANNOTATED_METRICS, QUALITY_METRICS
from shared.cli import parse_args
from shared.logger import get_logger
from shared.schema import DatasetSchema, TAG_GROUND_TRUTH


LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True)
    version: str = field(positional=True)
    run_dir: str = field(positional=True)


def run(args: Args):
    dataset = DatasetSchema.load_schema(args.dataset)
    version = dataset.get_version(args.version)
    config = BenchmarkConfig.load_config(args.baseline)
    LOG.info('Starting evaluation for dataset %s, version %s, baseline %s', args.dataset, args.version, args.baseline)

    context = EvaluationContext(
        dataset,
        version,
        config,
        Path(args.run_dir),
    )

    metrics = []
    if TAG_GROUND_TRUTH in dataset.tags:
        metrics.extend(ANNOTATED_METRICS)
    metrics.extend(QUALITY_METRICS)

    for metric_cls in metrics:
        metric = metric_cls(context)
        print(metric.evaluate())


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
