import os
from pathlib import Path

BASE_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

GLOBAL_CONFIG_PATH = BASE_PATH.joinpath('config')
CONFIG_DATASETS = GLOBAL_CONFIG_PATH.joinpath('datasets')
CONFIG_BENCHMARKS = GLOBAL_CONFIG_PATH.joinpath('benchmarks')

STORAGE_PATH = BASE_PATH.joinpath('storage')
TMP_PATH = STORAGE_PATH.joinpath('tmp')
CACHE_PATH = STORAGE_PATH.joinpath('cache')
REPORTS_PATH = STORAGE_PATH.joinpath('reports')

DATASETS_PATH = BASE_PATH.joinpath('datasets')
DATASETS_DATA = STORAGE_PATH.joinpath('datasets')
DATASETS_DATA_RAW = DATASETS_DATA.joinpath('raw')
DATASETS_DATA_PROCESSED = DATASETS_DATA.joinpath('processed')
DATASETS_DATA_EXPORT = DATASETS_DATA.joinpath('export')
DATASETS_DATA_VERSIONS = DATASETS_DATA.joinpath('versions')

BENCHMARKS_PATH = BASE_PATH.joinpath('benchmarks')
BENCHMARKS_OUTPUTS = BENCHMARKS_PATH.joinpath('outputs')
BENCHMARKS_LOGS = STORAGE_PATH.joinpath('logs')
BENCHMARKS_RESULTS = STORAGE_PATH.joinpath('results')

WANDB_PROJECT = 'Thesis'
WANDB_ENTITY = 'egordm'
