import shutil
import subprocess
import sys
from pathlib import Path
from natsort import natsorted

import argparse as argparse

parser = argparse.ArgumentParser(description='DYNAMO')
parser.add_argument('--input', type=str, default='/input', help='directory containing input edgelist file')
parser.add_argument('--output', type=str, default='/output/', help='output files dir')
parser.add_argument('--louvain', type=bool, default=False, action='store_true', help='run louvain')
args = parser.parse_args()

pwd_dir = Path(__file__).parent.absolute()
data_dir = pwd_dir.joinpath('data', 'input', 'ntwk')
data_dir.mkdir(exist_ok=True, parents=True)
input_dir = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

print('Copy input files to data directory')
snapshot_count = 0
for i, file in enumerate(sorted(input_dir.glob('*.edgelist'))):
    print('Processing file {}'.format(file))
    shutil.copy(file, data_dir.joinpath(str(i + 1)))
    snapshot_count += 1

print('Running DCD command')
cmd = [
    'java', '-jar', str(pwd_dir.joinpath('target/dynamic-1.0.jar')),
    'louvain' if args.louvain else 'dynamo', 'input', str(snapshot_count)
]
print(' '.join(cmd))
p = subprocess.Popen(
    cmd,
    shell=False,
    stdout=sys.stdout, stderr=sys.stderr,
)
p.wait()

print('Converting results to comlist format')
glob_str = 'runLouvain_*' if args.louvain else 'runDynamicModularity_*'
for i, file in enumerate(natsorted(data_dir.parent.glob('runDynamicModularity_*'), key=lambda x: str(x))):
    print('Processing file {}'.format(file))
    output_file = output_dir.joinpath(f'{str(i).zfill(2)}_snapshot.comlist')
    with output_file.open('w') as fout:
        with file.open() as fin:
            for i, line in enumerate(fin.readlines()):
                if len(line.strip()) == 0:
                    continue

                fout.write(f'{i}\t{line.strip()}\n')