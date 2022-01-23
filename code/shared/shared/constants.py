import os
from pathlib import Path

BASE_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

GLOBAL_CONFIG_PATH = BASE_PATH.joinpath('config')
CONFIG_DATASETS = GLOBAL_CONFIG_PATH.joinpath('datasets')
CONFIG_BENCHMARKS = GLOBAL_CONFIG_PATH.joinpath('benchmarks')

DATASETS_PATH = BASE_PATH.joinpath('datasets')
DATASETS_DATA_RAW = DATASETS_PATH.joinpath('data/raw')
DATASETS_DATA_PROCESSED = DATASETS_PATH.joinpath('data/processed')
DATASETS_DATA_EXPORT = DATASETS_PATH.joinpath('data/export')

BENCHMARKS_PATH = BASE_PATH.joinpath('benchmarks')
BENCHMARKS_OUTPUTS = BENCHMARKS_PATH.joinpath('outputs')
BENCHMARKS_LOGS = BENCHMARKS_OUTPUTS.joinpath('logs')
BENCHMARKS_RESULTS = BENCHMARKS_OUTPUTS.joinpath('results')
