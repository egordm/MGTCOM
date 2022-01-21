import argparse
import logging
import pathlib
import subprocess
import sys
import csv
from collections import defaultdict

# Determine the mode to use
MODES = [
    'louvain',
    'moses',
]
mode = None
for m in MODES:
    if m == sys.argv[1]:
        mode = m
        break

if mode is None:
    print("Please specify a mode: {}".format(MODES))
    sys.exit(1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='GreeneDCD')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose output')
parser.add_argument('--dynamic', action='store_true', default=False, help='calculate dynamic communities')
parser.add_argument('--input', type=str, default='/input', help='input directory')
parser.add_argument('--output', type=str, default='/output', help='output directory')

if mode == 'louvain':
    parser.add_argument('--weighted', type=str2bool, default=False, help='use weighted graph')
    parser.add_argument('--epsilon', type=float, default=0.001, help='epsilon for modularity optimization')
    parser.add_argument('--reuse_partition', type=str2bool, default=False,
                        help='reuse partition between consecutive timesteps')
    parser.add_argument('--level', type=int, default=-2,
                        help='level of coarseness to use from the hierarchy. -1 is the highest level, -2 is prompt at runtime')
elif mode == 'moses':
    pass

# GreeneDCD Arguments
parser.add_argument('--matching_threshold', type=float, default=0.1, help='(tracker) Matching threshold')
parser.add_argument('--death', type=int, default=3, help='(tracker) number of steps after which a dynamic community is declared dead')
parser.add_argument('--persist_threshold', type=int, default=1, help='(aggregator) Persist threshold')
parser.add_argument('--min_length', type=int, default=2, help='(aggregator) Minimum length of a track')
parser.add_argument('--max_step', type=int, default=-1, help='(aggregator) Maximum number of steps')

args = parser.parse_args(sys.argv[2:])

LOG = logging.getLogger('GreeneDCD')
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
LOG.addHandler(ch)

input_dir = pathlib.Path(args.input)
output_dir = pathlib.Path(args.output)
input_files = list(sorted(input_dir.glob('*.edgelist')))

if len(input_files) == 0:
    LOG.error('No input files found in {}'.format(input_dir))
    sys.exit(1)

tmp_dir = output_dir.joinpath('tmp')
tmp_dir.mkdir(exist_ok=True)

# Run community detection
community_files = []
if mode == 'louvain':
    LOG.info('Running Louvain Community Detection')
    snapshot_files = []
    for f in input_files:
        LOG.info(f'Converting input file {f} to binary format')
        output_file = tmp_dir.joinpath(f.with_suffix('.bin').name)
        p = subprocess.Popen(
            [
                './louvain-generic/convert',
                '-i', str(f),
                '-o', str(output_file),
                *(['-w', str(output_file.with_suffix('weights'))] if args.weighted else []),
            ],
            shell=False,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        p.wait()
        snapshot_files.append(output_file)

    tree_files = []
    for f in snapshot_files:
        LOG.info(f'Running Louvain on {f}')
        output_file = f.with_suffix('.tree')
        with open(str(output_file), 'w') as fout:
            p = subprocess.Popen(
                [
                    './louvain-generic/louvain',
                    str(f),
                    *(['-w', str(f.with_suffix('weights'))] if args.weighted else []),
                    '-q', '1',
                    '-e', str(args.epsilon),
                    *(['-v'] if args.verbose else []),
                    '-l', str(args.level if args.level >= 0 else '-1'),
                ],
                shell=False,
                stdout=fout, stderr=sys.stdout,
            )
            p.wait()
        tree_files.append(output_file)

    community_files = []
    level = args.level
    for f in tree_files:
        LOG.info(f'Extracting communities from {f}')
        p = subprocess.Popen(
            [
                './louvain-generic/hierarchy',
                str(f),
            ],
            shell=False,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        p.wait()

        if level == -2:
            if sys.stdout.isatty():
                inp = input('Enter level of coarseness to use [-1]: ')
            else:
                LOG.warning('Not running in a terminal, using default level of -1')
            level = int(inp) if inp else -1

        output_file = output_dir.joinpath(f.name).with_suffix('.comlist')
        with open(str(output_file), 'w') as fout:
            p = subprocess.Popen(
                [
                    './louvain-generic/hierarchy',
                    str(f),
                    *(['-l', str(level)] if level != -1 else ['-m']),
                ],
                shell=False,
                stdout=fout, stderr=sys.stdout,
            )
            p.wait()
        community_files.append(output_file)

    for f in community_files:
        LOG.info(f'Converting communities file {f} to row format')
        communities = defaultdict(list)
        with f.open('r') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node, community = line.split()
                    communities[int(community)].append(int(node))

        with f.open('w') as fout:
            for community, nodes in sorted(communities.items()):
                fout.write(' '.join(map(str, sorted(nodes))) + '\n')

elif mode == 'moses':
    LOG.info('Running Moses')
    community_files = []
    for f in input_files:
        LOG.info(f'Running Moses on {f}')
        output_file = output_dir.joinpath(f.with_suffix('.comlist').name)
        scores_file = tmp_dir.joinpath(f.with_suffix('.scores').name)
        p = subprocess.Popen(
            [
                './MOSES/moses',
                str(f),
                str(output_file),
                '--saveMOSESscores', str(scores_file),
            ],
            shell=False,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        p.wait()
        community_files.append(output_file)

if args.dynamic:
    N_TIMESTEPS = len(list(output_dir.glob('*.comlist')))

    LOG.info('Running dynamic community detection')
    output_file = tmp_dir.joinpath('communities')
    p = subprocess.Popen(
        [
            './tracker',
            '-t', str(args.matching_threshold),
            '-o', str(output_file),
            '--death', str(args.death),
            *list(sorted(map(str, output_dir.glob('*.comlist')))),
        ],
        shell=False,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    p.wait()

    print('Converitng dynamic community timeline to dynamic community format')
    community_timeline = []
    with tmp_dir.joinpath('communities.timeline').open('r') as f:
        for line in f.readlines():
            cid, timeline = line.strip()[1:].split(':')
            timeline_tokens = dict([tuple(map(int, token.split('='))) for token in timeline.split(',')])
            community_timeline.append((int(cid), timeline_tokens))

    with tmp_dir.joinpath('tracking.comtrack').open('w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['dcid', *(f'cid_t{i}' for i in range(N_TIMESTEPS))])
        for dcid, timeline in community_timeline:
            tsv_writer.writerow([dcid] + [timeline.get(i+1, '-') for i in range(N_TIMESTEPS)])

    print('Converting dynamic community timeline to pairwise matching format')
    match_list = set()
    for dcid, timeline in community_timeline:
        for i in range(N_TIMESTEPS - 1):
            if timeline.get(i+1, None) and timeline.get(i+2, None):
                match_list.add((i, i+1, timeline[i+1], timeline[i+2]))

    with output_dir.joinpath('tracking.tsv').open('w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['t_from', 't_to', 'cid_from', 'cid_to'])
        for t1, t2, c1, c2 in sorted(match_list):
            tsv_writer.writerow([t1, t2, c1 - 1, c2 - 1])

    LOG.info('Converting dynamic communities timeline to communities')
    p = subprocess.Popen(
        [
            './aggregator',
            '-i', str(tmp_dir.joinpath('communities.timeline')),
            '-o', str(tmp_dir.joinpath('aggregated.comlist')),
            '--persist', str(args.persist_threshold),
            '--max', str(args.max_step),
            '--length', str(args.min_length),
            *list(sorted(map(str, output_dir.glob('*.comlist')))),
        ],
        shell=False,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    p.wait()

LOG.info('Dynamic community detection complete')
