import argparse
import sys

import angel as a


is_angel = '--angel' in sys.argv

if is_angel:
    parser = argparse.ArgumentParser(description='ANGEL')
    parser.add_argument('--angel', action='store_true')
    parser.add_argument('--network_filename', type=str, default='/input/network.ncol', help='network filename')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold')
    parser.add_argument('--min_comsize', type=int, default=3, help='minimum community size')
    parser.add_argument('--out_filename', type=str, default='/output/communities.txt', help='output file path')
    args = parser.parse_args()

    an = a.Angel(
        network_filename=args.network_filename,
        threshold=args.threshold,
        min_com_size=args.min_comsize,
        outfile_path=args.out_filename
    )
    an.execute()
else:
    parser = argparse.ArgumentParser(description='ARCHANGEL')
    parser.add_argument('--angel', action='store_true')
    parser.add_argument('--network_filename', type=str, default='/input/network.ncol', help='network filename')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold')
    parser.add_argument('--match_threshold', type=float, default=0.4, help='match threshold')
    parser.add_argument('--min_comsize', type=int, default=3, help='minimum community size')
    parser.add_argument('--outfile_path', type=str, default='/output/', help='output file path')
    args = parser.parse_args()

    aa = a.ArchAngel(
        network_filename=args.network_filename,
        threshold=args.threshold,
        match_threshold=args.match_threshold,
        min_comsize=args.min_comsize,
        outfile_path=args.outfile_path
    )
    aa.execute()
