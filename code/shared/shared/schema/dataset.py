from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator

from astropy.io.misc import yaml
from simple_parsing import Serializable, field

from shared.constants import CONFIG_DATASETS, DATASETS_DATA_RAW, DATASETS_DATA_PROCESSED, \
    DATASETS_DATA_EXPORT, DATASETS_DATA_VERSIONS
from shared.string import to_identifier

TAG_DYNAMIC = 'dynamic'
TAG_STATIC = 'static'
TAG_GROUND_TRUTH = 'ground-truth'
TAG_OVERLAPPING = 'overlapping'
TAG_SYNTHETIC = 'synthetic'


class DatasetVersionType(Enum):
    EDGELIST_SNAPSHOTS = 'edgelist_snapshots'
    EDGELIST_STATIC = 'edgelist_static'

    def pretty(self) -> str:
        if self == DatasetVersionType.EDGELIST_SNAPSHOTS:
            return 'snapshots'
        elif self == DatasetVersionType.EDGELIST_STATIC:
            return 'static'


@dataclass
class DatasetVersionPart:
    path: Path

    def get_path(self):
        return self.path

    def exists(self):
        return self.path.exists()

    def snapshot_edgelist(self, i: int) -> Path:
        return self.path.joinpath(f'{str(i).zfill(2)}_snapshot.edgelist')

    def get_snapshot_edgelists(self) -> Iterator[Path]:
        return sorted(self.path.glob('*_snapshot.edgelist'))

    def snapshot_ground_truth(self, i: int) -> Path:
        return self.snapshot_edgelist(i).with_suffix('.comlist')

    def get_snapshot_ground_truths(self) -> Iterator[Path]:
        return map(lambda x: x.with_suffix('.comlist'), self.get_snapshot_edgelists())

    @property
    def static_edgelist(self) -> Path:
        return self.path.joinpath('static.edgelist')

    @property
    def static_ground_truth(self) -> Path:
        return self.static_edgelist.with_suffix('.comlist')

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
        return self.train

    @property
    def train(self):
        return DatasetVersionPart(self.get_path().joinpath('train'))

    def test_part(self) -> DatasetVersionPart:
        return self.test

    @property
    def test(self):
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

    def version(self, *args: str) -> Path:
        return DATASETS_DATA_VERSIONS.joinpath(self.name, *args)

    def version_str(self, *args: str) -> str:
        return str(self.version(*args))

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
            version._path = self.version(name)

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

    def is_synthetic(self) -> bool:
        return TAG_SYNTHETIC in self.tags
