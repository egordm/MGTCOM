from typing import List

import igraph as ig
import pandas as pd

from datasets.schema import DatasetSchema, EntitySchema


def count_types(schemas: List[EntitySchema]) -> pd.DataFrame:
    data = []

    for node_schema in schemas:
        df = node_schema.load_df()
        data.append({
            'type': str(node_schema),
            'count': len(df)
        })

    result = pd.DataFrame(data)
    result.set_index('type', inplace=True)
    return result

