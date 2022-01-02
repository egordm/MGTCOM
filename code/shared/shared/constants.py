import os
from pathlib import Path

BASE_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

GLOBAL_CONFIG_PATH = BASE_PATH.joinpath('config')

DATASETS_PATH = BASE_PATH.joinpath('datasets')
DATASETS_DATA_RAW = DATASETS_PATH.joinpath('data/raw')
DATASETS_DATA_PROCESSED = DATASETS_PATH.joinpath('data/processed')
DATASETS_DATA_EXPORT = DATASETS_PATH.joinpath('data/export')


class DatasetPath:
    def __init__(self, name):
        self.name = name

    def raw(self, *args: str) -> Path:
        return DATASETS_DATA_RAW.joinpath(self.name, *args)

    def raw_str(self, *args: str) -> str:
        return str(self.raw(*args))

    def processed(self, *args: str) -> Path:
        return DATASETS_DATA_PROCESSED.joinpath(self.name, *args)

    def processed_str(self, *args: str) -> str:
        return str(self.processed(*args))

    def export(self, *args: str) -> Path:
        return DATASETS_DATA_EXPORT.joinpath(self.name, *args)

    def export_str(self, *args: str) -> str:
        return str(self.export(*args))

    def __str__(self):
        return self.name
