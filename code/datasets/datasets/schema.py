import os.path
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, ClassVar, Callable, Any, Iterator, Union

import pandas as pd
from simple_parsing.helpers import Serializable

from shared.constants import DATASETS_PATH, DatasetPath


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
    timestamp: Optional[bool] = False

    IGNORE_PROPS = ["ignore", "label", "timestamp"]

    def is_id(self):
        return self.name == 'id'

    def is_src(self):
        return self.name == 'src'

    def is_dst(self):
        return self.name == 'dst'

    def is_array(self):
        return '[]' in self.type


@dataclass(order=True)
class HasPropertiesMixin:
    properties: List[Property]

    def iter_properties(self, predicate: Callable[[Property], bool] = None) -> Iterator[Property]:
        for prop in self.properties:
            if predicate is None or predicate(prop):
                yield prop

    def get_label(self) -> Optional[Property]:
        return next(self.iter_properties(lambda prop: prop.label), None)

    def get_timestamp(self) -> Optional[Property]:
        return next(self.iter_properties(lambda prop: prop.timestamp), None)


@dataclass(order=True)
class LoadableDataframeMixin:
    path: str

    def get_path(self):
        return os.path.join(DATASETS_PATH, self.path)

    def load_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.get_path())


@dataclass(order=True)
class NodeSchema(Serializable, Mergeable, HasPropertiesMixin, LoadableDataframeMixin):
    label: str
    path: str
    properties: List[Property]

    MERGE_FN = {
        'properties': merge_list_by_key(lambda p: p.name)
    }

    def get_type(self) -> str:
        return self.label

    def __str__(self):
        return f'{self.label}'


@dataclass(order=True)
class EdgeSchema(Serializable, Mergeable, HasPropertiesMixin, LoadableDataframeMixin):
    type: str
    source: str
    target: str
    path: str
    properties: List[Property]
    directed: bool = True

    IGNORE_PROPS = ["directed"]
    MERGE_FN = {
        'properties': merge_list_by_key(lambda p: p.name)
    }

    def get_type(self) -> str:
        return self.type

    def __str__(self):
        template = '({})-[{}]->({})' if self.directed else '({})-[{}]-({})'
        return template.format(self.source, self.type, self.target)


SCHEMA_DIR = os.path.join(DATASETS_PATH, 'schemas')

EntitySchema = Union[NodeSchema, EdgeSchema]


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

    def all_properties(self) -> Iterator[Property]:
        for node in self.nodes:
            yield from node.properties
        for edge in self.edges:
            yield from edge.properties

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

    def paths(self) -> DatasetPath:
        return DatasetPath(self.name)

    def get_node_schema(self, label: str) -> NodeSchema:
        return next(filter(lambda node: node.label == label, self.nodes), None)

    def get_edge_schema(self, type: str) -> EdgeSchema:
        return next(filter(lambda edge: edge.type == type, self.edges), None)

    def get_node_types(self):
        return [node.get_type() for node in self.nodes]

    def get_edge_types(self):
        return [edge.get_type() for edge in self.edges]
