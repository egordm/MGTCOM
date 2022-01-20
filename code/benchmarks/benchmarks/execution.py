import datetime as dt
import logging
import os
import subprocess

from typeguard import check_type

from benchmarks.benchmarks.config import BenchmarkConfig, ParameterConfig, RawParameterValue
from datasets.schema import DatasetSchema
from shared.constants import BENCHMARKS_RESULTS, BASE_PATH
from shared.logger import get_logger, get_logpipe

LOG = get_logger('Executor')


def params_to_args(params: ParameterConfig):
    result = []
    for k, v in params.items():
        if check_type(k, v, RawParameterValue):
            raise Exception("Only raw parameter values are supported")

        value = str(v) if not isinstance(v, bool) else str(v).lower()
        result.append(f'--{k}={value}')
    return result


def execute_benchmark(
        config: BenchmarkConfig,
        params: ParameterConfig,
        dataset: DatasetSchema,
        dataset_version: str,
        prefix: str,
):
    run_name = f'{prefix}_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    dataset_version = dataset.get_version(dataset_version)
    input_dir = dataset_version.get_path()
    output_dir = BENCHMARKS_RESULTS.joinpath(config.name, dataset.name, run_name)
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
