import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional

import wandb
from simple_parsing import field

from benchmarks.config import BenchmarkConfig
from benchmarks.scripts import execute, execute_evaluate
from shared.cli import parse_args
from shared.constants import BENCHMARKS_LOGS, WANDB_PROJECT
from shared.logger import get_logger
from shared.schema import DatasetSchema
from shared.threading import AsyncModelTimeout

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: str = field(positional=True, help="dataset name")
    version: Optional[str] = field(default=None, help="dataset version name")


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
        **({
            'metric': baseline.get_metric(dataset.name),
        } if baseline.get_metric(dataset.name) else {})
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
            run_model_with_parameters = lambda: execute_evaluate.run(
                execute.Args(
                    baseline=args.baseline,
                    dataset=args.dataset,
                    version=args.version,
                    run_name=run_name,
                ),
                params=params,
            )
            async_model = AsyncModelTimeout(run_model_with_parameters, baseline.get_timeout(dataset.name))
            success, result = async_model.run()
            if success:
                LOG.info(f"Evaluation of {run_name} finished with result {result}")
                wandb.log(result)
            else:
                LOG.error(f"Evaluation of {run_name} timed out")
                wandb.log({
                    'error': 'timeout',
                    'timeout': baseline.get_timeout(dataset.name),
                })

    wandb.agent(sweep_id, function=runner, count=baseline.get_run_count(dataset.name))


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    if args.version:
        LOG.info(f"Running {args.baseline} on {args.dataset}:{args.version}")
        run(args)
    else:
        baseline = BenchmarkConfig.load_config(args.baseline)
        if not baseline.datasets.get(args.dataset, None):
            raise ValueError(f"Dataset {args.dataset} not found in baseline {args.baseline}")

        for version in baseline.datasets[args.dataset].versions:
            LOG.info(f"Running {args.baseline} on {args.dataset}:{args.version}")
            args.version = version
            run(args)
