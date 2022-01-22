from dataclasses import dataclass
from typing import Union, List

from simple_parsing import field

from benchmarks.benchmarks.config import BenchmarkConfig
from shared.cli import parse_args
from shared.config import ConnectionConfig


@dataclass
class Args:
    config: str = field(positional=True, help="config name")
    run: str = field(positional=True, help="run name")


args: Args = parse_args(Args)[0]

global_config = ConnectionConfig.load_config()
config = BenchmarkConfig.from_name(args.config)
