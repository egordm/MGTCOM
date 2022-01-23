import itertools as it
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.evaluation import EvaluationContext, ANNOTATED_METRICS, QUALITY_METRICS
from shared.cli import parse_args
from shared.schema import DatasetSchema


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True)
    version: str = field(positional=True)
    output_dir: str = field(positional=True)


args: Args = parse_args(Args)[0]

dataset = DatasetSchema.load_schema(args.dataset)
version = dataset.get_version(args.version)
config = BenchmarkConfig.load_config(args.baseline)

context = EvaluationContext(
    dataset,
    version,
    config,
    Path(args.output_dir),
)

for metric_cls in it.chain(ANNOTATED_METRICS, QUALITY_METRICS):
    metric = metric_cls(context)
    print(metric.evaluate())
