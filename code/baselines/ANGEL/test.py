import argparse

import tiles as t

parser = argparse.ArgumentParser(description='ANGEL')
parser.add_argument('--input', type=str, default='/input', help='directory containing input edgelist file')
parser.add_argument('--output', type=str, default='/output', help='output file path')
args = parser.parse_args()

# aa = a.ArchAngel(
#     network_filename=str(args.input),
#     outfile_path=str(args.output),
#     neighborhood_size=2
# )
# aa.execute()

tiles = t.TILES(
    filename=str(args.input),
    path=str('output_test'),
)
