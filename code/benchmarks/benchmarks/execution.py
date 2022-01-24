import datetime as dt
import logging
import os
import subprocess
from typing import Dict, Any, Optional

from benchmarks.benchmarks.config import BenchmarkConfig
from shared.constants import BENCHMARKS_RESULTS, BASE_PATH
from shared.logger import get_logger, get_logpipe
from shared.schema import DatasetSchema

LOG = get_logger(os.path.basename(__file__))


def params_to_args(params: Dict[str, Any]):
    result = []
    for k, v in params.items():
        if not isinstance(v, (int, float, str, bool)):
            raise Exception("Only raw parameter values are supported")

        value = str(v) if not isinstance(v, bool) else str(v).lower()
        result.append(f'--{k}={value}')
    return result


def execute_benchmark(
        config: BenchmarkConfig,
        params: Dict[str, Any],
        dataset: DatasetSchema,
        dataset_version: str,
        run_name: Optional[str] = None,
):
    if run_name is None:
        run_name = f'run_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{dataset}_{dataset_version}'

    version = dataset.get_version(dataset_version)
    input_dir = version.train.get_path()
    output_dir = BENCHMARKS_RESULTS.joinpath(config.name, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        *config.execution.entrypoint,
        *params_to_args(params),
        '--input', str(config.execution.input_path(input_dir)),
        '--output', str(config.execution.output_path(output_dir)),
    ]
    LOG.debug(f'Executing command: {" ".join(command)}')

    if config.execution.entrypoint:
        with get_logpipe('Baseline') as outpipe:
            with get_logpipe('Baseline', logging.ERROR) as errorpipe:
                p = subprocess.Popen(
                    command,
                    shell=False,
                    stdout=outpipe, stderr=errorpipe,
                    cwd=str(BASE_PATH.joinpath(config.execution.cwd)),
                    env={
                        **os.environ,
                        **config.execution.env,
                    }
                )
                p.wait()
    else:
        raise Exception("No entrypoint defined")
