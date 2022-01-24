import datetime as dt
import os
from dataclasses import dataclass

from simple_parsing import field

import wandb
from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.scripts import execute, evaluate, execute_evaluate
from shared.cli import parse_args
from shared.config import ConnectionConfig
from shared.constants import BENCHMARKS_LOGS, WANDB_PROJECT, BENCHMARKS_RESULTS
from shared.logger import get_logger
from shared.schema import DatasetSchema

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True, help="dataset name")
    version: str = field(positional=True, help="dataset version name")
    run_count: int = field(default=1, help="number of runs")


def run(args: Args):
    baseline = BenchmarkConfig.load_config(args.baseline)
    dataset = DatasetSchema.load_schema(args.dataset)

    tags = [
        'baseline',
        args.baseline,
        args.dataset,
        f'{args.dataset}:{args.version}',
        *{*baseline.tags, *dataset.tags}
    ]

    os.environ['WANDB_DIR'] = str(BENCHMARKS_LOGS)
    os.environ['WANDB_TAGS'] = ','.join(tags)

    current_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_id = wandb.sweep({
        "name": f"sweep-{current_dt}-{args.baseline}-{args.dataset}:{args.version}",
        "method": "random",
        "parameters": {
            **baseline.get_params(dataset.name).to_dict(),
            "baseline": {
                'value': args.baseline
            },
            "dataset": {
                'value': args.dataset
            },
            "version": {
                'value': args.version
            },
        },
    }, project=WANDB_PROJECT)

    def runner():
        current_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{current_dt}-{args.baseline}-{args.dataset}:{args.version}"

        with wandb.init(
                name=run_name,
        ) as run:
            config = wandb.config
            params = {
                param: config[param]
                for param in baseline.get_params(dataset.name).to_dict().keys()
                if param in config
            }
            result = execute_evaluate.run(
                execute.Args(
                    baseline=args.baseline,
                    dataset=args.dataset,
                    version=args.version,
                    run_name=run_name,
                ),
                params=params,
            )
            LOG.info(f"Evaluation of {run_name} finished with result {result}")
            wandb.log(result)

    wandb.agent(sweep_id, function=runner, count=args.run_count)


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
