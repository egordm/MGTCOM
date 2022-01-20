from pypher import Pypher, __

from datasets.neo4j import query_snapshot
from datasets.schema import DatasetSchema
from shared.constants import DatasetPath

# DATASET = DatasetPath('star-wars')
DATASET = DatasetPath('social-distancing-student')
schema = DatasetSchema.load_schema(DATASET.name)

range_start = 0
range_end = 10



print(query_snapshot(
schema, range_start, range_end, raw_vars=True,
))

# MATCH (a)-[r]-(b)
# WHERE
#  ((
#     r:ANSWERED_QUESTION AND
#     r.timestamp > datetime({year: 2011}) AND
#     r.timestamp <= datetime({year: 2012})
#
# ) OR (
#     r:COMMENTED_ON_QUESTION AND
#     r.timestamp > datetime({year: 2012}) AND
#     r.timestamp <= datetime({year: 2013})
# )) AND id(b) = 98220 AND id(a) =  91828
# RETURN r LIMIT 10
