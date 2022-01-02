import os.path
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, ClassVar, Callable, Any

from simple_parsing.helpers import Serializable

from shared.constants import DATASETS_PATH


@dataclass
class Mergeable:
    """
    Mixin to make a class mergable.
    """

    IGNORE_PROPS: ClassVar[str] = []
    MERGE_FN: ClassVar[Dict[str, Callable[[Any, Any], Any]]] = {}

    def __merge_field(self, other, field):
        self_value = getattr(self, field.name)
        other_value = getattr(other, field.name)

        if field.name in self.MERGE_FN:
            return self.MERGE_FN[field.name](self_value, other_value)
        else:
            return merge_by_value(self_value, other_value)

    def merge(self, other):
        data = {
            field.name: self.__merge_field(other, field)
            if field.name not in self.IGNORE_PROPS else getattr(self, field.name)
            for field in fields(self)
        }

        return type(self)(**data)


def merge_by_value(self_value, other_value):
    if isinstance(self_value, Mergeable):
        return self_value.merge(other_value)
    else:
        return other_value


def merge_list_by_key(key_fn: Callable[[Any], str]):
    def merge_fn(self, other):
        self_items = {key_fn(item): item for item in self}
        other_items = {key_fn(item): item for item in other}

        return [
            merge_by_value(self_items[key], other_item) if key in self_items else other_item
            for (key, other_item) in other_items.items()
        ]

    return merge_fn


@dataclass(order=True)
class Property(Serializable, Mergeable):
    name: str
    type: str
    ignore: Optional[bool] = False
    label: Optional[bool] = False

    IGNORE_PROPS = ["ignore", "label"]

    def is_id(self):
        return self.name == 'id'

    def is_src(self):
        return self.name == 'src'

    def is_dst(self):
        return self.name == 'dst'

    def is_array(self):
        return '[]' in self.type


@dataclass(order=True)
class NodeSchema(Serializable, Mergeable):
    label: str
    path: str
    properties: List[Property]

    MERGE_FN = {
        'properties': merge_list_by_key(lambda p: p.name)
    }


@dataclass(order=True)
class EdgeSchema(Serializable, Mergeable):
    type: str
    source: str
    target: str
    path: str
    properties: List[Property]

    MERGE_FN = {
        'properties': merge_list_by_key(lambda p: p.name)
    }


SCHEMA_DIR = os.path.join(DATASETS_PATH, 'schema')


@dataclass(order=True)
class DatasetSchema(Serializable, Mergeable):
    name: str
    prefix: str
    database: str
    description: str = ''
    nodes: List[NodeSchema] = field(default_factory=list)
    edges: List[EdgeSchema] = field(default_factory=list)

    IGNORE_PROPS = ["description"]
    MERGE_FN = {
        'nodes': merge_list_by_key(lambda node: node.label),
        'edges': merge_list_by_key(lambda edge: edge.type)
    }

    @staticmethod
    def load_schema(name: str) -> 'DatasetSchema':
        """
        Loads the schema from the given path.
        """
        path = os.path.join(SCHEMA_DIR, f'{name}.yml')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Schema file not found: {path}')
        return DatasetSchema.load(path)

    def save_schema(self) -> None:
        """
        Saves the schema to the given path.
        """
        self.save(os.path.join(SCHEMA_DIR, f'{self.name}.yml'), sort_keys=False)
