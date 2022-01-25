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
    index = int(file.name.split('.')[1]) - 1
    out_file = output_dir.joinpath(str(index).zfill(2) + '_snapshot.comlist')

    with out_file.open('w') as fout:
        with file.open('r') as fin:
            for line in fin.readlines():
                node, community = line.split()
                fout.write(f'{int(node) - 1}\t{int(community) - 1}\n')
