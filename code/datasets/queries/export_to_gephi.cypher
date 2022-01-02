match path = ()-[r]->()
WITH path LIMIT 1000
with collect(path) as paths
call apoc.gephi.add(null,'workspace1', paths) yield nodes, relationships, time
return nodes, relationships, time
