import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Any, Type

from simple_parsing import Serializable, field

from shared.constants import BENCHMARKS_CONFIGS, BASE_PATH, DATASETS_DATA_EXPORT, BENCHMARKS_OUTPUTS
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


@dataclass
class ExecutionConfig(Serializable):
    cwd: pathlib.Path
    entrypoint: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = field(default_factory=dict)
    containerized: bool = False

    def input_path(self, path: pathlib.Path) -> pathlib.Path:
        if self.containerized:
            return pathlib.Path('/input') / path.relative_to(DATASETS_DATA_EXPORT)
        else:
            return BASE_PATH.joinpath(path)

    def output_path(self, path: pathlib.Path) -> pathlib.Path:
        if self.containerized:
            return pathlib.Path('/output') / path.relative_to(BENCHMARKS_OUTPUTS)
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
    def from_name(name) -> 'BenchmarkConfig':
        path = BENCHMARKS_CONFIGS.joinpath(f'{name}.yml')
        if not path.exists():
            raise FileNotFoundError(f'{path} does not exist')
        return BenchmarkConfig.load(path)

# def
