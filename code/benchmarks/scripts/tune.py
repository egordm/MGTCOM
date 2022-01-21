from dataclasses import dataclass
from typing import Union, List

import wandb
from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from shared.cli import parse_args
from shared.config import ConnectionConfig


@dataclass
class Args:
    config: str = field(positional=True, help="config name")
    datasets: Union[str, List[str]] = field(default='all', alias='-d')


args: Args = parse_args(Args)[0]

global_config = ConnectionConfig.load_config()
config = BenchmarkConfig.from_name(args.config)

# global_config.wandb.open()

print(
    args
)
print(
    config
)



sweep_config = {
    "name": "my-sweep",
    "method": "random",
    "parameters": config.get_params('star-wars').to_dict(),
}
sweep_id = wandb.sweep(sweep_config)


def run():
    with wandb.init() as run:
        config = wandb.config
        # execute_benchmark(
        #     config,
        #     config.get_params('star-wars'),
        #     DatasetSchema.load_schema('star-wars'),
        #     # 'snapshots_k=5',
        #     'static',
        #     'test'
        # )
        u = 0


count = 1
wandb.agent(sweep_id, function=run, count=count)
