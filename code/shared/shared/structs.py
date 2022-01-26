from typing import Any


def filter_none_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def filter_list_none_values(l: list) -> list:
    return [v for v in l if v is not None]


def tuple_to_dict(data: tuple, keys) -> dict:
    return dict(zip(keys, data))


def filter_none_values_recursive(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: filter_none_values_recursive(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [filter_none_values_recursive(v) for v in d if v is not None]
    else:
        return d


def dict_deep_merge(a: dict, b: dict) -> dict:
    """
    Recursively merge dicts a and b.
    Keys in b take precedence over keys in a.
    """
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict):
            result[k] = dict_deep_merge(result[k], v)
        else:
            result[k] = v
    return result
