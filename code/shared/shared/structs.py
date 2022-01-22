from typing import Any


def filter_none_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def tuple_to_dict(data: tuple, keys) -> dict:
    return dict(zip(keys, data))

def filter_none_values_recursive(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: filter_none_values_recursive(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [filter_none_values_recursive(v) for v in d if v is not None]
    else:
        return d