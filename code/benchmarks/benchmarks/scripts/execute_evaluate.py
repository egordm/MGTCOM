import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.scripts import execute, evaluate
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

    if args.run_name is None:
        current_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.run_name = f"{current_dt}-{args.baseline}-{args.dataset}:{args.version}"

    if params is None:
        params = baseline.get_params(dataset.name).to_simple_dict()

    LOG.info(f"Running {args.run_name} with params {params}")
    execute.run(
        execute.Args(
            baseline=args.baseline,
            dataset=args.dataset,
            version=args.version,
            run_name=args.run_name,
        ),
        params=params,
    )
    LOG.info(f"Running evaluation of {args.run_name}")
    result = evaluate.run(
        evaluate.Args(
            baseline=args.baseline,
            dataset=args.dataset,
            version=args.version,
            run_name=args.run_name,
        )
    )
    LOG.info(f"Evaluation of {args.run_name} finished with result {result}")
    return result


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
