import itertools as it
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Type, Union

import inflection
import pyspark.sql.types as T
import yaml
from simple_parsing import Serializable as Serializable, field

from shared.schema import DatasetSchema
from shared.structs import filter_none_values_recursive

EntityType = str
PropertyName = str


class DTypeAtomic(Enum):
    STRING = 'string'
    INT = 'int'
    FLOAT = 'float'
    BOOL = 'boolean'
    DATETIME = 'datetime'
    DATE = 'date'

    @classmethod
    def from_spark(cls, stype: T.DataType) -> 'DTypeAtomic':
        if isinstance(stype, T.StringType):
            return DTypeAtomic.STRING
        elif isinstance(stype, T.IntegerType):
            return DTypeAtomic.INT
        elif isinstance(stype, T.LongType):
            return DTypeAtomic.INT
        elif isinstance(stype, T.FloatType):
            return DTypeAtomic.FLOAT
        elif isinstance(stype, T.DoubleType):
            return DTypeAtomic.FLOAT
        elif isinstance(stype, T.BooleanType):
            return DTypeAtomic.BOOL
        elif isinstance(stype, T.TimestampType):
            return DTypeAtomic.DATETIME
        elif isinstance(stype, T.DateType):
            return DTypeAtomic.DATE
        else:
            raise NotImplementedError(f'{stype}')


@dataclass
class DType(Serializable):
    atomic: DTypeAtomic
    array: bool = False

    def to_dict(self, dict_factory: Type[Dict] = dict, recurse: bool = True) -> Dict:
        return str(self)

    @classmethod
    def from_dict(cls: Type['DType'], obj: Dict, drop_extra_fields: bool = None) -> 'DType':
        return DType(DTypeAtomic(obj)) if '[]' not in obj else DType(DTypeAtomic(obj[:-2]), True)

    def __str__(self):
        return f'{self.atomic.value}' if not self.array else f'{self.atomic.value}[]'

    @classmethod
    def from_spark(cls, stype: T.DataType) -> 'DType':
        if isinstance(stype, T.ArrayType):
            return DType(DTypeAtomic.from_spark(stype.elementType), True)
        else:
            return DType(DTypeAtomic.from_spark(stype), False)


@dataclass
class GraphProperty(Serializable):
    _name: str = field(init=False, default=None, to_dict=False)
    dtype: DType

    @property
    def name(self):
        return self._name


@dataclass
class DynamicConfig(Serializable):
    timestamp: PropertyName
    interaction: bool = False


@dataclass
class EntitySchema(Serializable):
    _type: Optional[EntityType] = field(init=False, default=None, to_dict=False)
    _schema: Optional['GraphSchema'] = field(init=False, default=None, to_dict=False)
    label: PropertyName = None
    properties: Dict[PropertyName, GraphProperty] = field(default_factory=dict)
    dynamic: Optional[DynamicConfig] = None

    def __post_init__(self):
        for k, v in self.properties.items():
            v._name = k

    def add_property(self, name: PropertyName, prop: GraphProperty) -> None:
        self.properties[name] = prop
        prop._name = name

    def get_type(self) -> EntityType:
        return self._type

    def is_dynamic(self) -> bool:
        return self.dynamic is not None

    @abstractmethod
    def get_path(self) -> Path:
        pass

    @classmethod
    def from_spark(
            cls,
            schema: T.StructType,
            label: Optional[PropertyName] = None,
            timestamp: Optional[PropertyName] = None,
            interaction: Optional[bool] = False,
            **kwargs,
    ) -> 'EntitySchema':
        properties = {
            name: GraphProperty(DType.from_spark(stype.dataType))
            for name, stype in zip(schema.names, schema.fields)
        }
        label_prop = label if label else None
        if label_prop and label_prop not in properties:
            raise ValueError(f'Label property {label_prop} not found in schema')

        dynamic = DynamicConfig(timestamp, interaction) if timestamp else None
        if dynamic and dynamic.timestamp not in properties:
            raise ValueError(f'Timestamp property {dynamic.timestamp} not found in schema')

        return cls(label=label_prop, properties=properties, dynamic=dynamic, **kwargs)


@dataclass
class NodeSchema(EntitySchema):
    def __str__(self):
        return f'({self.get_type()})'

    def get_path(self) -> Path:
        return self._schema.get_path().joinpath(f'nodes_{self.get_type()}')


@dataclass
class EdgeSchema(EntitySchema):
    source_type: EntityType = None
    target_type: EntityType = None
    directed: bool = False

    def __str__(self):
        template = '({})-[{}]->({})' if self.directed else '({})-[{}]-({})'
        return template.format(self.source_type, self._type, self.target_type)

    def get_path(self) -> Path:
        return self._schema.get_path().joinpath(f'edges_{self.get_type()}')

    @classmethod
    def from_spark(
            cls,
            schema: T.StructType,
            label: Optional[PropertyName] = None,
            timestamp: Optional[PropertyName] = None,
            interaction: Optional[bool] = False,
            source_type: EntityType = None,
            target_type: EntityType = None,
            directed: bool = False,
            **kwargs,
    ) -> 'EdgeSchema':
        return super(EdgeSchema, cls).from_spark(
            schema, label, timestamp, interaction,
            source_type=source_type,
            target_type=target_type,
            directed=directed,
            **kwargs,
        )


@dataclass
class GraphSchema(Serializable):
    _path: Optional[Path] = field(init=False, default=None, to_dict=False)
    nodes: Dict[EntityType, NodeSchema] = field(default_factory=dict)
    edges: Dict[EntityType, EdgeSchema] = field(default_factory=dict)

    def __post_init__(self):
        for entity_type, entity_schema in it.chain(self.nodes.items(), self.edges.items()):
            entity_schema._schema = self
            entity_schema._type = entity_type

    def add_node_schema(self, type_name: EntityType, schema: NodeSchema) -> 'GraphSchema':
        if inflection.camelize(type_name, True) != type_name:
            raise ValueError(f'Invalid type name: {type_name}. Must be PascalCase.')

        self.nodes[type_name] = schema
        schema._schema = self
        schema._type = type_name

        return self

    def get_node_schema(self, node_type: EntityType) -> NodeSchema:
        return self.nodes[node_type]

    def add_edge_schema(self, type_name: EntityType, schema: EdgeSchema) -> 'GraphSchema':
        if inflection.underscore(type_name).upper() != type_name:
            raise ValueError(f'Invalid type name: {type_name}. Must be CONST_CASE.')

        self.edges[type_name] = schema
        schema._schema = self
        schema._type = type_name

        return self

    def get_edge_schema(self, edge_type: EntityType) -> EdgeSchema:
        return self.edges[edge_type]

    def get_node_types(self) -> list:
        return list(self.nodes.keys())

    def get_edge_types(self) -> list:
        return list(self.edges.keys())

    def get_path(self) -> Path:
        if self._path is None:
            raise ValueError('No path set for schema')
        return self._path

    def set_path(self, path: Path) -> None:
        self._path = path

    def is_dynamic(self) -> bool:
        return any(schema.dynamic is not None for schema in it.chain(self.nodes.values(), self.edges.values()))

    def is_node_temporal(self) -> bool:
        return any(schema.dynamic is not None for schema in self.nodes.values())

    def is_edge_temporal(self) -> bool:
        return any(schema.dynamic is not None for schema in self.edges.values())

    def is_node_interaction(self) -> bool:
        return any(
            schema.dynamic.interaction
            for schema in it.chain(self.nodes.values(), self.edges.values())
            if schema.dynamic is not None
        )

    @classmethod
    def load_schema(cls: Type['GraphSchema'], path: Union[str, Path], **kwargs) -> 'GraphSchema':
        path = Path(path) if isinstance(path, str) else path
        data = yaml.safe_load(path.joinpath('schema.yaml').read_text(), **kwargs)
        result = cls.from_dict(data)
        result._path = path
        return result

    def save_schema(self, path: Optional[Union[str, Path]] = None, **kwargs) -> 'GraphSchema':
        if path is None:
            path = self.get_path()

        path.mkdir(parents=True, exist_ok=True)
        self._path = Path(path) if isinstance(path, str) else path

        for schema in it.chain(self.nodes.values(), self.edges.values()):
            if not schema.get_path().exists():
                raise ValueError(f'Path for subschema not exist: {schema.get_path()}')

        data = filter_none_values_recursive(self.to_dict())
        self._path.joinpath('schema.yaml').write_text(yaml.dump(data, **kwargs))
        return self

    @classmethod
    def from_dataset(cls, schema: DatasetSchema) -> 'GraphSchema':
        return GraphSchema.load_schema(schema.processed())

