import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from simple_parsing import field

from benchmarks.config import BenchmarkConfig
from benchmarks.execution import execute_benchmark
from shared.cli import parse_args
from shared.logger import get_logger
from shared.schema import DatasetSchema

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True, help="dataset name")
    version: str = field(positional=True, help="dataset version name")
    run_name: Optional[str] = field(default=None, help="run name")


def run(args: Args, params: Optional[Dict[str, Any]] = None):
    baseline = BenchmarkConfig.load_config(args.baseline)
    dataset = DatasetSchema.load_schema(args.dataset)

    if params is None:
        params = baseline.get_params(str(dataset)).to_simple_dict()

    execute_benchmark(
        baseline,
        params,
        dataset,
        args.version,
        args.run_name,
    )


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
