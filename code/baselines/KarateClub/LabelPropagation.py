import argparse
import logging
import pathlib
from collections import defaultdict

import networkx as nx
from karateclub import LabelPropagation

LOG = logging.getLogger('LabelPropagation')
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
LOG.addHandler(ch)

parser = argparse.ArgumentParser(description="LabelPropagation")
parser.add_argument('--input', type=str, default='/input/graph.edgelist', help='input graph edgelist file')
parser.add_argument('--output', type=str, default='/output', help='output directory')
parser.add_argument('--iterations', type=int, default=100, help='iterations')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

output_dir = pathlib.Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# Read graph
LOG.info('Reading graph...')
G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.Graph())
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')
G.to_undirected()

# Run GEMSEC
LOG.info('Running LabelPropagation...')
model = LabelPropagation(
    seed=args.seed,
    iterations=args.iterations,
)
model.fit(G)
memberships = model.get_memberships()

# Save communities
LOG.info('Saving communities...')
communities = defaultdict(list)
for node, community in memberships.items():
    communities[community].append(node)

with output_dir.joinpath('communities.txt').open('w') as fout:
    for community, nodes in communities.items():
        fout.write(' '.join(map(str, nodes)) + '\n')
