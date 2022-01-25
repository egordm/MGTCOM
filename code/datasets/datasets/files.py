from typing import Iterator

from shared.constants import CONFIG_DATASETS


def get_dataset_names() -> Iterator[str]:
    for file in CONFIG_DATASETS.glob('*.yaml'):
        yield file.stem

