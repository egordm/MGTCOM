from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from astropy.io.misc import yaml
from simple_parsing import Serializable, field

from shared.constants import CONFIG_DATASETS, DATASETS_DATA_RAW, DATASETS_DATA_PROCESSED, \
    DATASETS_DATA_EXPORT
from shared.string import to_identifier


class DatasetVersionType(Enum):
    SNAPSHOTS = 'snapshots'
    STATIC = 'static'


@dataclass
class DatasetVersionPart:
    path: Path

    def get_path(self):
        return self.path

    def exists(self):
        return self.path.exists()

    @property
    def snapshots(self) -> Path:
        return self.path.joinpath('snapshots')

    @property
    def static(self) -> Path:
        return self.path.joinpath('static.edgelist')

    @property
    def ground_truth(self) -> Path:
        return self.path.joinpath('ground_truth.comlist')

    @property
    def nodemapping(self) -> Path:
        return self.path.joinpath('nodemapping.tsv')


@dataclass
class DatasetVersion(Serializable):
    _path: Path = field(init=False, default=None, to_dict=False)
    type: DatasetVersionType = field(decoding_fn=lambda x: DatasetVersionType(x), encoding_fn=lambda x: x.value)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_path(self) -> Path:
        return self._path

    def train_part(self) -> DatasetVersionPart:
        return DatasetVersionPart(self.get_path().joinpath('train'))

    def test_part(self) -> DatasetVersionPart:
        return DatasetVersionPart(self.get_path().joinpath('test'))

    def get_param(self, name: str, default=None) -> Optional[Any]:
        return self.parameters.get(name, default)


@dataclass
class DatasetPath:
    name: str

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


@dataclass
class DatasetSchema(DatasetPath, Serializable):
    name: str
    database: str
    description: str = ''
    versions: Dict[str, DatasetVersion] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        for name, version in self.versions.items():
            version._path = self.export('versions').joinpath(name)

    @classmethod
    def load_schema(cls, name, **kwargs) -> 'DatasetSchema':
        path = CONFIG_DATASETS.joinpath(f'{name}.yaml')
        if path.exists():
            data = yaml.load(path.read_text(), **kwargs)
            result = cls.from_dict(data)
        else:
            result = cls(name=name, database=to_identifier(name).replace('_', '-'))

        return result

    def save_schema(self, **kwargs):
        path = CONFIG_DATASETS.joinpath(f'{self.name}.yaml')
        yaml.dump(self.to_dict(), path.open('w'), **kwargs)

    def get_version(self, version: str) -> DatasetVersion:
        if version not in self.versions:
            raise ValueError(f'Version {version} not found in dataset {self.name}')

        return self.versions[version]
