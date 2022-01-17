import os
import subprocess
import sys
import datetime as dt

from typeguard import check_type

from benchmarks.benchmarks.config import BenchmarkConfig, ParameterConfig, RawParameterValue
from datasets.schema import DatasetSchema
from shared.constants import BENCHMARKS_RESULTS, BASE_PATH, BENCHMARKS_OUTPUTS, DATASETS_DATA_EXPORT
from shared.logger import get_logger

LOG = get_logger('Executor')


def params_to_args(params: ParameterConfig):
    result = []
    for k, v in params.items():
        if check_type(k, v, RawParameterValue):
            raise Exception("Only raw parameter values are supported")

        result.extend([
            f'--{k}',
            str(v),
        ])
    return result


def execute_benchmark(
        config: BenchmarkConfig,
        params: ParameterConfig,
        dataset: DatasetSchema,
        prefix: str,
):
    if config.input.type == 'edgelist':
        input_dir = dataset.paths().export('edgelist')
    else:
        raise Exception(f'Unsupported input type: {config.input.type}')

    run_name = f'{prefix}_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = BENCHMARKS_RESULTS.joinpath(config.name, dataset.name, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.execution.entrypoint:
        if config.input.input_prefix:
            input_dir_rel = os.path.join(config.input.input_prefix, str(input_dir.relative_to(DATASETS_DATA_EXPORT)))
        else:
            input_dir_rel = str(input_dir)

        if config.input.output_prefix:
            output_dir_rel = os.path.join(config.input.output_prefix, str(output_dir.relative_to(BENCHMARKS_OUTPUTS)))
        else:
            output_dir_rel = str(output_dir)

        p = subprocess.Popen(
            [
                *config.execution.entrypoint,
                *params_to_args(params),
                '--input', str(input_dir_rel),
                '--output', str(output_dir_rel),
            ],
            shell=False,
            stdout=sys.stdout, stderr=sys.stderr,
            cwd=str(BASE_PATH.joinpath(config.execution.cwd)),
            env={
                **os.environ,
                **config.execution.env,
            }
        )
        p.wait()
    else:
        raise Exception("No entrypoint defined")
