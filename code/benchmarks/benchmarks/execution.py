import datetime as dt
import logging
import os
import subprocess
from threading import Timer
from typing import Dict, Any, Optional

from benchmarks.config import BenchmarkConfig
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
        baseline: BenchmarkConfig,
        params: Dict[str, Any],
        dataset: DatasetSchema,
        dataset_version: str,
        run_name: Optional[str] = None,
        timeout: Optional[int] = None,
):
    if run_name is None:
        run_name = f'run_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{dataset}_{dataset_version}'

    version = dataset.get_version(dataset_version)
    input_dir = version.train.get_path()
    output_dir = BENCHMARKS_RESULTS.joinpath(baseline.name, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        *baseline.execution.entrypoint,
        *params_to_args(params),
        '--input', str(baseline.execution.input_path(input_dir)),
        '--output', str(baseline.execution.output_path(output_dir)),
    ]
    LOG.debug(f'Executing command: {" ".join(command)}')

    if baseline.execution.entrypoint:
        with get_logpipe('Baseline') as outpipe:
            with get_logpipe('Baseline', logging.ERROR) as errorpipe:
                p = subprocess.Popen(
                    command,
                    shell=False,
                    stdout=outpipe, stderr=errorpipe,
                    cwd=str(BASE_PATH.joinpath(baseline.execution.cwd)),
                    env={
                        **os.environ,
                        **baseline.execution.env,
                    }
                )

                timed_out = False

                def timeout_fn():
                    nonlocal timed_out
                    p.kill()
                    timed_out = True

                if timeout:
                    timer = Timer(timeout, timeout_fn)

                try:
                    if timeout:
                        timer.start()
                    p.wait()
                finally:
                    if timeout:
                        timer.cancel()

                if timed_out:
                    raise TimeoutError(f"Timeout of {timeout} seconds exceeded")

                if p.returncode != 0:
                    raise Exception(f'Benchmark failed with exit code {p.returncode}')
    else:
        raise Exception("No entrypoint defined")
