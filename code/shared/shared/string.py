import re


def to_identifier(s: str) -> str:
    s = re.sub('[- ]', '_', s)
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)
