def filter_none_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def tuple_to_dict(data: tuple, keys) -> dict:
    return dict(zip(keys, data))