from dataclasses import dataclass
from typing import Union, List

import wandb
from simple_parsing import field, ArgumentParser

from benchmarks.benchmarks.config import BenchmarkConfig
from benchmarks.benchmarks.execution import execute_benchmark
from datasets.schema import DatasetSchema
from shared.config import ConnectionConfig


@dataclass
class Args:
    config: str = field(positional=True, help="config name")
    dataset: Union[str, List[str]] = field('all', alias='-d')


parser = ArgumentParser()
parser.add_arguments(Args, dest="args")
args: Args = parser.parse_args().args

global_config = ConnectionConfig.load_config()
config = BenchmarkConfig.from_name(args.config)

# global_config.wandb.open()

print(
    args
)
print(
    config
)


execute_benchmark(
    config,
    config.get_params('ucidata-zachary'),
    DatasetSchema.load_schema('ucidata-zachary'),
    'test'
)