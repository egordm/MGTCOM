import argparse
import logging
import pathlib
from collections import defaultdict

import networkx as nx
import numpy as np
from karateclub import GEMSEC

LOG = logging.getLogger('GEMSEC')
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
LOG.addHandler(ch)

parser = argparse.ArgumentParser(description="GEMSEC")
parser.add_argument('--input', type=str, default='/input', help='driectory containing input edgelist file')
parser.add_argument('--output', type=str, default='/output', help='output directory')
parser.add_argument('--walk_number', type=int, default=5, help='number of random walks')
parser.add_argument('--walk_length', type=int, default=80, help='length of random walks')
parser.add_argument('--dimensions', type=int, default=32, help='embedding dimensions')
parser.add_argument('--negative_samples', type=int, default=5, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=5, help='window size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--clusters', type=int, default=10, help='number of clusters')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

output_dir = pathlib.Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = output_dir.joinpath('tmp')
tmp_dir.mkdir(parents=True, exist_ok=True)

input_dir = pathlib.Path(args.input)
input_file = next(input_dir.glob('*.edgelist'))

# Read graph
LOG.info('Reading graph...')
G = nx.read_edgelist(str(input_file), nodetype=int, create_using=nx.Graph())
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')
G.to_undirected()

# Run GEMSEC
LOG.info('Running GEMSEC...')
model = GEMSEC(
        walk_number=args.walk_number,
        walk_length=args.walk_length,
        dimensions=args.dimensions,
        negative_samples=args.negative_samples,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        clusters=args.clusters,
        gamma=args.gamma,
        seed=args.seed
)
model.fit(G)
embedding = model.get_embedding()
memberships = model.get_memberships()

# Save embeddings
LOG.info('Saving embeddings...')
np.savez(tmp_dir.joinpath('embeddings.npz'), embedding=embedding)

# Save communities
LOG.info('Saving communities...')
communities = defaultdict(list)
for node, community in memberships.items():
    communities[community].append(node)

with output_dir.joinpath('default.coms').open('w') as fout:
    for community, nodes in communities.items():
        fout.write(' '.join(map(str, nodes)) + '\n')