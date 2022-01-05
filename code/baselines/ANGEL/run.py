import argparse
import sys
import pathlib
from collections import defaultdict

import angel as a


is_angel = '--angel' in sys.argv

if is_angel:
    parser = argparse.ArgumentParser(description='ANGEL')
    parser.add_argument('--angel', action='store_true')
    parser.add_argument('--input', type=str, default='/input/network.ncol', help='network filename')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold')
    parser.add_argument('--min_comsize', type=int, default=3, help='minimum community size')
    parser.add_argument('--output', type=str, default='/output/communities.txt', help='output file path')
    args = parser.parse_args()

    an = a.Angel(
        network_filename=args.input,
        threshold=args.threshold,
        min_com_size=args.min_comsize,
        outfile_path=args.output
    )
    an.execute()
    outputs = [pathlib.Path(args.output)]
else:
    parser = argparse.ArgumentParser(description='ARCHANGEL')
    parser.add_argument('--angel', action='store_true')
    parser.add_argument('--input', type=str, default='/input', help='Directory of input files')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold')
    parser.add_argument('--match_threshold', type=float, default=0.4, help='match threshold')
    parser.add_argument('--min_comsize', type=int, default=3, help='minimum community size')
    parser.add_argument('--output', type=str, default='/output/', help='output files dir')
    args = parser.parse_args()

    print('Converting snapshot networks to temporal network...')
    input_dir = pathlib.Path(args.input)
    output_file = pathlib.Path(args.output).joinpath('temporal_network.ncol')
    with output_file.open('w') as wf:
        for i, in_f in enumerate(sorted(map(str, input_dir.glob('*.txt')))):
            edges = []
            with open(in_f, 'r') as rf:
                for line in rf.readlines():
                    u, v = line.split()
                    edges.append((u, v, i))

            for u, v, i in edges:
                wf.write('{}\t{}\t{}\n'.format(u, v, i))

    print('Running ANGEL...')
    aa = a.ArchAngel(
        network_filename=str(output_file),
        threshold=args.threshold,
        match_threshold=args.match_threshold,
        min_comsize=args.min_comsize,
        outfile_path=args.output
    )
    aa.execute()
    outputs = list(sorted(map(str, pathlib.Path(args.output).glob('*_coms_*.txt'))))


print('Converting communities to list format...')
for i, output in enumerate(outputs):
    output = pathlib.Path(output)
    communities = list()
    with output.open('r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            ci, nodes = line.strip().split('\t')
            communities.append(eval(nodes))

    with output.with_name(str(i).zfill(2) + '.communities.txt').open('w') as f:
        for ci, nodes in enumerate(communities):
            f.write(' '.join(nodes) + '\n')
