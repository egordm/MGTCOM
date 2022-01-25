import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from simple_parsing import field

from benchmarks.config import BenchmarkConfig
from benchmarks.execution import execute_benchmark
from shared.cli import parse_args
from shared.logger import get_logger
from shared.schema import DatasetSchema
from shared.threading import AsyncModelTimeout

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True, help="dataset name")
    version: str = field(positional=True, help="dataset version name")
    run_name: Optional[str] = field(default=None, help="run name")
    timeout: Optional[int] = field(default=None, help="timeout in seconds")


def run(args: Args, params: Optional[Dict[str, Any]] = None):
    baseline = BenchmarkConfig.load_config(args.baseline)
    dataset = DatasetSchema.load_schema(args.dataset)

    if params is None:
        params = baseline.get_params(str(dataset)).to_simple_dict()

    run_model_with_parameters = lambda: execute_benchmark(
        baseline,
        params,
        dataset,
        args.version,
        args.run_name,
    )

    if args.timeout is not None and args.timeout > 0:
        async_model = AsyncModelTimeout(run_model_with_parameters, baseline.get_timeout(dataset.name))
        success, result = async_model.run()
        if not success:
            raise TimeoutError(f"Timeout of {args.timeout} seconds exceeded")
    else:
        run_model_with_parameters()


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
