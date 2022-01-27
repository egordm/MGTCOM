import copy
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
from shared.exceptions import NoCommunitiesFoundError
from shared.logger import get_logger
from shared.schema import DatasetSchema
from shared.structs import filter_list_none_values

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    baseline: str = field(positional=True, help="baseline config name")
    dataset: Optional[str] = field(alias='d', default=None, help="dataset name")
    version: Optional[str] = field(alias='v', default=None, help="dataset version name")


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
            "timeout": {
                'value': baseline.get_timeout(dataset.name)
            }
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
            try:
                result = execute_evaluate.run(
                    execute.Args(
                        baseline=args.baseline,
                        dataset=args.dataset,
                        version=args.version,
                        run_name=run_name,
                        timeout=baseline.get_timeout(dataset.name)
                    ),
                    params=params,
                )
                success = True
                LOG.info(f"Evaluation of {run_name} finished with result {result}")
            except TimeoutError as e:
                LOG.error(f"TimeoutError: {e}")
                wandb.log({
                    'error': 'timeout',
                    'timeout': baseline.get_timeout(dataset.name),
                })
                success = False
            except NoCommunitiesFoundError as e:
                LOG.error(f"NoCommunitiesFoundError: {e}")
                wandb.log({
                    'error': 'no_communities',
                })
                success = False
            except Exception as e:
                LOG.error(f"Exception: {e}")
                wandb.finish(exit_code=1)
                raise e

            if success:
                LOG.info(f"Evaluation of {run_name} finished with result {result}")
                wandb.log(result)
                wandb.finish(exit_code=0)
            else:
                LOG.error(f"Evaluation of {run_name} failed.")
                wandb.finish(exit_code=1)

    wandb.agent(sweep_id, function=runner, count=baseline.get_run_count(dataset.name))
    wandb.finish(exit_code=0)


if __name__ == "__main__":
    args: Args = parse_args(Args)[0]
    baseline = BenchmarkConfig.load_config(args.baseline)

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = baseline.datasets.keys()

    for dataset in datasets:
        if args.version:
            versions = [args.version]
        else:
            if not baseline.datasets.get(dataset, None):
                raise ValueError(f"Dataset {dataset} not found in baseline {args.baseline}")
            versions = baseline.datasets[dataset].versions

        for version in versions:
            new_args = copy.deepcopy(args)
            new_args.dataset = dataset
            new_args.version = version

            LOG.info(f"Running {new_args.baseline} on {new_args.dataset}:{new_args.version}")
            run(new_args)
