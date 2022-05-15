import os
from dataclasses import dataclass
from pathlib import Path

BASE_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

CONFIG_PATH = BASE_PATH / 'config'

STORAGE_PATH = BASE_PATH / 'storage'
TMP_PATH = STORAGE_PATH / 'tmp'
CACHE_PATH = STORAGE_PATH / 'cache'
DATASET_PATH = STORAGE_PATH / 'datasets'
RESULTS_PATH = STORAGE_PATH / 'results'
EXPORTS_PATH = STORAGE_PATH / 'exports'
OUTPUTS_PATH = STORAGE_PATH / 'outputs'



@dataclass
class DatasetPath:
    name: str

    def raw(self, *args: str) -> Path:
        return DATASET_PATH.joinpath('raw', self.name, *args)

    def raw_str(self, *args: str) -> str:
        return str(self.raw(*args))

    def processed(self, *args: str) -> Path:
        return DATASET_PATH.joinpath('processed', self.name, *args)

    def processed_str(self, *args: str) -> str:
        return str(self.processed(*args))

    def export(self, *args: str) -> Path:
        return DATASET_PATH.joinpath('export', self.name, *args)

    def export_str(self, *args: str) -> str:
        return str(self.export(*args))

    def __str__(self):
        return self.name
