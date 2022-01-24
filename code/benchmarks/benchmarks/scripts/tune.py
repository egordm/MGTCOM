import datetime as dt
import os
from dataclasses import dataclass

import wandb
from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.scripts import execute
from shared.cli import parse_args
from shared.config import ConnectionConfig
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
    global_config = ConnectionConfig.load_config()
    global_config.wandb.open(config={
        "baseline": args.baseline,
        "dataset": args.dataset,
        "version": args.version,
    })

    baseline = BenchmarkConfig.load_config(args.baseline)
    dataset = DatasetSchema.load_schema(args.dataset)

    current_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_config = {
        "name": f"sweep-{args.baseline}-{args.dataset}:{args.version}-{current_dt}",
        "method": "random",
        "parameters": baseline.get_params(dataset.name).to_dict(),
    }
    sweep_id = wandb.sweep(sweep_config)

    def runner():
        with wandb.init() as run:
            config = wandb.config
            u = 0
            execute.run(
                execute.Args(
                    baseline=args.baseline,
                    dataset=args.dataset,
                    version=args.version,
                ),

            )

    wandb.agent(sweep_id, function=runner, count=args.run_count)


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    run(args)
