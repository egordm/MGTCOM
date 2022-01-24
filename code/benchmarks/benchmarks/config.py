import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Any, Type

from simple_parsing import Serializable, field

from shared.constants import CONFIG_BENCHMARKS, BASE_PATH, BENCHMARKS_RESULTS, \
    DATASETS_DATA_VERSIONS
from shared.structs import filter_none_values

RawParameterValue = Any


@dataclass
class TunedParameterValue(Serializable):
    value: Optional[RawParameterValue] = None
    values: Optional[List[RawParameterValue]] = None
    min: Optional[RawParameterValue] = None
    max: Optional[RawParameterValue] = None

    def to_dict(self, dict_factory: Type[Dict] = dict, recurse: bool = True) -> Dict:
        return filter_none_values(super().to_dict(dict_factory, recurse))

    def to_simple(self):
        if self.values is not None:
            return self.values[0]
        elif self.value is not None:
            return self.value
        else:
            return self.min


ParameterValue = Union[RawParameterValue, TunedParameterValue]


# ParameterConfig = Dict[str, ParameterValue]

class ParameterConfig(dict, Serializable):
    @classmethod
    def from_dict(cls: Type['ParameterConfig'], obj: Dict, drop_extra_fields: bool = None) -> 'ParameterConfig':
        return ParameterConfig(**{
            key: TunedParameterValue.from_dict(value) if isinstance(value, dict) else TunedParameterValue(value=value)
            for key, value in obj.items()
        })

    def to_dict(self, dict_factory: Type[Dict] = dict, recurse: bool = True) -> Dict:
        return {
            key: value.to_dict()
            for key, value in self.items()
        }

    def to_simple_dict(self) -> Dict:
        return {
            key: value.to_simple()
            for key, value in self.items()
        }


@dataclass
class ExecutionConfig(Serializable):
    cwd: pathlib.Path
    entrypoint: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = field(default_factory=dict)
    containerized: bool = False

    def input_path(self, path: pathlib.Path) -> pathlib.Path:
        if self.containerized:
            return pathlib.Path('/input') / path.relative_to(DATASETS_DATA_VERSIONS)
        else:
            return BASE_PATH.joinpath(path)

    def output_path(self, path: pathlib.Path) -> pathlib.Path:
        if self.containerized:
            return pathlib.Path('/output') / path.relative_to(BENCHMARKS_RESULTS)
        else:
            return BASE_PATH.joinpath(path)


@dataclass
class DatasetRunConfig(Serializable):
    parameters: ParameterConfig = field(default_factory=dict)
    versions: List[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig(Serializable):
    name: str
    description: str
    execution: ExecutionConfig
    tags: List[str] = field(default_factory=list)
    parameters: ParameterConfig = field(default_factory=dict)
    datasets: Dict[str, DatasetRunConfig] = field(default_factory=dict)

    def get_params(self, dataset_name: str) -> ParameterConfig:
        return ParameterConfig(**{
            **self.parameters,
            **self.datasets.get(dataset_name, DatasetRunConfig()).parameters
        })

    @staticmethod
    def load_config(name) -> 'BenchmarkConfig':
        path = CONFIG_BENCHMARKS.joinpath(f'{name}.yml')
        if not path.exists():
            raise FileNotFoundError(f'{path} does not exist')
        return BenchmarkConfig.load(path)
