import argparse
import os
import pathlib
from collections import defaultdict

import matlab.engine

parser = argparse.ArgumentParser(description='ESPRA')
parser.add_argument('--input', type=str, default='Directory containing input snapshots')
parser.add_argument('--output', type=str, default='Directory to save output snapshots')
parser.add_argument('--alpha', type=float, default=0.8, help='Parameter for balancing the current clustering (=1) and historical influence (=0)')
parser.add_argument('--beta', type=float, default=0.5, help='Parameter for trading off the emphasis between the structural perturbation and topological similarity')

input_dir = pathlib.Path(parser.parse_args().input)
output_dir = pathlib.Path(parser.parse_args().output)
output_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = output_dir.joinpath('tmp')
tmp_dir.mkdir(parents=True, exist_ok=True)

print('Starting MATLAB engine...')
eng = matlab.engine.start_matlab()
eng.cd(os.path.abspath(os.path.dirname(__file__)))

print('Running ESPRA...')
eng.runESPRA(
    list(sorted(map(str, input_dir.glob('*.edgelist')))),
    str(tmp_dir).rstrip('/') + '/',
    parser.parse_args().alpha,
    parser.parse_args().beta,
    nargout=0
)

print('Converting output snapshots...')
for file in tmp_dir.glob('dynamic.*.communities.txt'):
    with file.open('r') as f:
        communities = defaultdict(list)
        for line in f.readlines():
            node, community = line.split()
            communities[int(community)].append(int(node))

    index = file.name.split('.')[1]
    community_count = max(communities.keys())
    with output_dir.joinpath(str(index).zfill(2) + '_snapshot.coms').open('w') as f:
        for community in range(community_count):
            nodes = communities.get(community + 1, [])
            f.write(' '.join(map(str, nodes)) + '\n')
