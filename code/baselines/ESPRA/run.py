import argparse
import os
import pathlib

import matlab.engine

parser = argparse.ArgumentParser(description='ESPRA')
parser.add_argument('--input', type=str, default='Directory containing input snapshots')
parser.add_argument('--output', type=str, default='Directory to save output snapshots')
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.5)

input_dir = pathlib.Path(parser.parse_args().input)
output_dir = pathlib.Path(parser.parse_args().output)
output_dir.mkdir(parents=True, exist_ok=True)

print('Starting MATLAB engine...')
eng = matlab.engine.start_matlab()
eng.cd(os.path.abspath(os.path.dirname(__file__)))

print('Running ESPRA...')
eng.runESPRA(
    list(sorted(map(str, input_dir.glob('*.txt')))),
    str(output_dir).rstrip('/') + '/',
    parser.parse_args().alpha,
    parser.parse_args().beta,
    nargout=0
)
