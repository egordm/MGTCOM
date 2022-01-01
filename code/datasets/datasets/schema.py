import os.path
from dataclasses import dataclass, field
from typing import List, Optional

from simple_parsing.helpers import Serializable

from shared.constants import DATASETS_PATH


@dataclass(order=True)
class Property(Serializable):
    name: str
    type: str
    ignore: Optional[bool] = False

    def is_id(self):
        return self.name == 'id'

    def is_src(self):
        return self.name == 'src'

    def is_dst(self):
        return self.name == 'dst'

    def is_array(self):
        return '[]' in self.type


@dataclass(order=True)
class NodeSchema(Serializable):
    label: str
    id: str
    path: str
    properties: List[Property]


@dataclass(order=True)
class EdgeSchema(Serializable):
    type: str
    source: str
    target: str
    path: str
    properties: List[Property]


@dataclass(order=True)
class DatasetSchema(Serializable):
    name: str
    prefix: str
    database: str
    description: str = ''
    nodes: List[NodeSchema] = field(default_factory=list)
    edges: List[EdgeSchema] = field(default_factory=list)


SCHEMA_DIR = os.path.join(DATASETS_PATH, 'schema')


def load_schema(name: str) -> DatasetSchema:
    """
    Loads the schema from the given path.
    """
    path = os.path.join(SCHEMA_DIR, f'{name}.yml')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Schema file not found: {path}')
    return DatasetSchema.load(path)


def save_schema(schema: DatasetSchema) -> None:
    """
    Saves the schema to the given path.
    """
    schema.save(os.path.join(SCHEMA_DIR, f'{schema.name}.yml'), sort_keys=False)
