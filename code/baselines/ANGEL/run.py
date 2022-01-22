import argparse
import re
import sys
import pathlib
from collections import defaultdict
import csv

import angel as a


is_angel = '--angel' in sys.argv

if is_angel:
    parser = argparse.ArgumentParser(description='ANGEL')
    parser.add_argument('--angel', action='store_true')
    parser.add_argument('--input', type=str, default='/input', help='directory containing input edgelist file')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold')
    parser.add_argument('--min_comsize', type=int, default=3, help='minimum community size')
    parser.add_argument('--output', type=str, default='/output', help='output file path')
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    tmp_dir = output_dir.joinpath('tmp')
    tmp_dir.mkdir(exist_ok=True)
    an = a.Angel(
        network_filename=str(input_dir.joinpath('default.edgelist')),
        threshold=args.threshold,
        min_comsize=args.min_comsize,
        outfile_name=tmp_dir.joinpath('default.out'),
    )
    an.execute()
    outputs = [tmp_dir.joinpath('default.out')]
else:
    parser = argparse.ArgumentParser(description='ARCHANGEL')
    parser.add_argument('--angel', action='store_true')
    parser.add_argument('--input', type=str, default='/input', help='Directory of input edgelist files')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold')
    parser.add_argument('--match_threshold', type=float, default=0.4, help='match threshold')
    parser.add_argument('--min_comsize', type=int, default=3, help='minimum community size')
    parser.add_argument('--output', type=str, default='/output/', help='output files dir')
    args = parser.parse_args()

    print('Converting snapshot networks to temporal network...')
    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    tmp_dir = pathlib.Path(args.output).joinpath('tmp')
    tmp_dir.mkdir(exist_ok=True)
    output_file = tmp_dir.joinpath('temporal_network.tsv')
    with output_file.open('w') as wf:
        for i, in_f in enumerate(sorted(map(str, input_dir.glob('*.edgelist')))):
            edges = []
            with open(in_f, 'r') as rf:
                for line in rf.readlines():
                    u, v = re.split(r'\s|\t', line.strip())
                    edges.append((u.strip(), v.strip(), i))

            for u, v, i in edges:
                wf.write('{}\t{}\t{}\n'.format(u, v, i))

    print('Running ANGEL...')
    aa = a.ArchAngel(
        network_filename=str(output_file),
        threshold=args.threshold,
        match_threshold=args.match_threshold,
        min_comsize=args.min_comsize,
        outfile_path=str(tmp_dir.joinpath('output_'))
    )
    aa.execute()
    outputs = list(sorted(map(str, tmp_dir.glob('*_coms_*.txt'))))

    print('Converting community tracking results...')
    with output_dir.joinpath('tracking.tsv').open('w') as wf:
        tsv_writer = csv.writer(wf, delimiter='\t')
        tsv_writer.writerow(['t_from', 't_to', 'cid_from', 'cid_to'])
        with tmp_dir.joinpath('output_ArchAngel_coms_ct_matches.csv').open('r', newline='') as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                tsv_writer.writerow([row['snapshot_from'], row['snapshot_to'], row['cid_from'], row['cid_to']])

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

    with output_dir.joinpath(str(i).zfill(2) + '.coms').open('w') as f:
        for ci, nodes in enumerate(communities):
            f.write(' '.join(nodes) + '\n')
