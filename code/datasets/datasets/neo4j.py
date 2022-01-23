import datetime as dt
from typing import Any

from pypher import Pypher, __
from pypher.builder import create_statement

from shared.schema import DatasetSchema


def value_to_neo4j(x):
    # Kinda injection hazard, use with caution
    if x is None:
        return 'null'
    elif isinstance(x, str):
        return '"{}"'.format(x)
    elif isinstance(x, bool):
        return 'true' if x else 'false'
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, float):
        return str(x)
    elif isinstance(x, dt.datetime):
        return 'datetime("{}")'.format(x.isoformat())
    else:
        raise ValueError('Unsupported type: {}'.format(type(x)))


create_statement('RawValue', {'name': ''})


def query_snapshot(
        schema: DatasetSchema,
        range_start: Any,
        range_end: Any,
        raw_vars: bool = False,
):
    if raw_vars:
        range_start = __.RawValue(range_start)
        range_end = __.RawValue(range_end)

    q = Pypher()
    q.MATCH.node('a').rel_out('r').node('b')

    node_start_conditionals = []
    node_end_conditionals = []
    edge_conditionals = []

    for node_schema in schema.nodes:
        def build_cond(node_ident: str):
            c = [
                getattr(__, node_ident).label(node_schema.label)
            ]
            if node_schema.is_temporal():
                t_prop = node_schema.get_timestamp()
                if node_schema.interaction:
                    c.append(getattr(__, node_ident).property(t_prop.name) >= range_start)
                c.append(getattr(__, node_ident).property(t_prop.name) <= range_end)
            return __.ConditionalAND(*c)

        node_start_conditionals.append(build_cond('a'))
        node_end_conditionals.append(build_cond('b'))

    for edge_schema in schema.edges:
        c = [
            getattr(__, 'r').label(edge_schema.type)
        ]
        if edge_schema.is_temporal():
            t_prop = edge_schema.get_timestamp()
            if edge_schema.interaction:
                c.append(getattr(__, 'r').property(t_prop.name) >= range_start)
            c.append(getattr(__, 'r').property(t_prop.name) <= range_end)

        edge_conditionals.append(__.ConditionalAND(*c))

    q.WHERE.ConditionalAND(
        __.ConditionalOR(*node_start_conditionals),
        __.ConditionalOR(*node_end_conditionals),
        __.ConditionalOR(*edge_conditionals)
    )

    return q


def export_to_csv(
        query: Pypher,
        output_path: str,
        export_options: dict
):
    export_options = {
        'delim': ' ',
        'quotes': False,
        'separateHeader': True,
        **export_options,
    }

    q = Pypher()
    q.WITH(str(query)).alias('query')
    q.CALL(__.func('apoc.export.csv.query', __.query, output_path, export_options))
    q.YIELD('file', 'source', 'format', 'nodes', 'relationships', 'properties', 'time', 'rows', 'batchSize', 'batches',
            'done', 'data')
    q.RETURN('file', 'source', 'format', 'nodes', 'relationships', 'properties', 'time', 'rows', 'batchSize', 'batches',
             'done', 'data')

    q.bind_params(query.bound_params)
    query.safely_stringify_for_pudb()
