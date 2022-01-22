import argparse
import logging
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np



parser = argparse.ArgumentParser(description='CommunityGAN')
# -nt:Number of threads for parallelization(default: 16)=
# -c:The number of communities to detect (-1 detect automatically)(default: 500)=
# -mc:Minimum number of communities to try(default: 5)=
# -xc:Maximum number of communities to try(default: 500)=
# -nc:How many trials for the number of communities(default: 10)=
# -sa:Alpha for backtracking line search(default: 0.05)=
# -sb:Beta for backtracking line search(default: 0.1)=
# -st:Allow reference between two same time node or not (0: don't allow, 1: allow)(default: 0)=
# -woe:Disable Eta or not (0: enable eta, 1: disable eta, 2: symmetric eta)(default: 1)=
# -se:same Eta or not (0: different eta, 1: same eta)(default: 1)=
# -mi:Maximum number of update iteration(default: 500)=
# -si:How many iterations for once save(default: 5000)=
# -rsi:How many iterations for once negative sampling(default: 10)=
# -sa:Zero Threshold for F and eta(default: 0.0001)=
# -lnf:Remain only largest how many elements for F(default: 0)=

parser.add_argument('--input_file', type=str, default='/input/dataset.edges', help='input file')
parser.add_argument('--ground_truth', type=str, default=None, help='input ground truth')
parser.add_argument('--output_dir', type=str, default='/output', help='output directory')
parser.add_argument('--n_communities', type=int, default=-1, help='number of communities')
parser.add_argument('--min_communities', type=int, default=5, help='minimum number of communities')
parser.add_argument('--max_communities', type=int, default=500, help='maximum number of communities')
parser.add_argument('--n_trials', type=int, default=10, help='number of trials for the number of communities')
parser.add_argument('--alpha', type=float, default=0.05, help='alpha for backtracking line search')
parser.add_argument('--beta', type=float, default=0.1, help='beta for backtracking line search')
parser.add_argument('--allow_reference', type=int, default=0, help='allow reference between two same time node or not')
parser.add_argument('--disable_eta', type=int, default=1, help='disable Eta or not')
parser.add_argument('--same_eta', type=int, default=1, help='same Eta or not')
parser.add_argument('--max_iter', type=int, default=500, help='maximum number of update iteration')
parser.add_argument('--save_iter', type=int, default=5000, help='how many iterations for once save')
parser.add_argument('--sample_iter', type=int, default=10, help='how many iterations for once negative sampling')
parser.add_argument('--zero_threshold', type=float, default=0.0001, help='zero threshold for F and eta')
parser.add_argument('--remain_largest', type=int, default=0, help='remain only largest how many elements for F')

# # training settings
# self.motif_size = 3  # number of nodes in a motif
# self.batch_size_gen = 64  # batch size for the generator
# self.batch_size_dis = 64  # batch size for the discriminator
# self.n_sample_gen = 5  # number of samples for the generator
# self.n_sample_dis = 5  # number of samples for the discriminator
# self.lr_gen = 1e-3  # learning rate for the generator
# self.lr_dis = 1e-3  # learning rate for the discriminator
# self.n_epochs = 10  # number of outer loops
# self.n_epochs_gen = 3  # number of inner loops for the generator
# self.n_epochs_dis = 3  # number of inner loops for the discriminator
# self.gen_interval = self.n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
# self.dis_interval = self.n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
# self.update_ratio = 1  # updating ratio when choose the trees
# self.max_value = 1000  # max value in embedding matrix
#
# # model saving
# self.load_model = True  # whether loading existing model for initialization
# self.save_steps = 10
#
# # other hyper-parameters
# self.n_emb = 100
# self.num_threads = 16
# self.window_size = 5
#
# # application and dataset settings
# self.app = "community_detection"
# self.dataset = "com-amazon"
parser.add_argument('--motif_size', type=int, default=3, help='number of nodes in a motif')
parser.add_argument('--batch_size_gen', type=int, default=64, help='batch size for the generator')
parser.add_argument('--batch_size_dis', type=int, default=64, help='batch size for the discriminator')
parser.add_argument('--n_sample_gen', type=int, default=5, help='number of samples for the generator')
parser.add_argument('--n_sample_dis', type=int, default=5, help='number of samples for the discriminator')
parser.add_argument('--lr_gen', type=float, default=1e-3, help='learning rate for the generator')
parser.add_argument('--lr_dis', type=float, default=1e-3, help='learning rate for the discriminator')
parser.add_argument('--n_epochs', type=int, default=10, help='number of outer loops')
parser.add_argument('--n_epochs_gen', type=int, default=3, help='number of inner loops for the generator')
parser.add_argument('--n_epochs_dis', type=int, default=3, help='number of inner loops for the discriminator')
parser.add_argument('--gen_interval', type=int, default=3, help='sample new nodes for the generator for every gen_interval iterations')
parser.add_argument('--dis_interval', type=int, default=3, help='sample new nodes for the discriminator for every dis_interval iterations')
parser.add_argument('--update_ratio', type=int, default=1, help='updating ratio when choose the trees')
parser.add_argument('--max_value', type=float, default=1000, help='max value in embedding matrix')
parser.add_argument('--load_model', type=bool, default=True, help='whether loading existing model for initialization')
parser.add_argument('--save_steps', type=int, default=10, help='how many iterations for once save')
parser.add_argument('--n_emb', type=int, default=100, help='number of embedding dimension')
parser.add_argument('--num_threads', type=int, default=16, help='number of threads')
parser.add_argument('--window_size', type=int, default=5, help='window size for negative sampling')
parser.add_argument('--app', type=str, default="community_detection", help='application')
parser.add_argument('--overlap_threshold', type=float, default=1.0, help='Threshold for overlapping community extraction')
# parser.add_argument('--dataset', type=str, default="com-amazon", help='dataset')
parser.add_argument('--dataset', type=str, default="zz-top", help='dataset')

args = parser.parse_args()

# Setup Log
LOG = logging.getLogger('CommunityGAN')
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
LOG.addHandler(ch)


# Preflight checks
input_file = pathlib.Path(args.input_file)
if not input_file.exists():
    LOG.error("Input file does not exist: %s", input_file)
    exit(1)
if input_file.is_dir():
    input_file = next(input_file.glob('*.edgelist'))

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = output_dir.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

# Make sure nodes start with 0 index
LOG.info("Checking input file: %s", input_file)
edges = []
with input_file.open('r') as f:
    min_node_id = 999999
    for line in f.readlines():
        u, v = map(int, re.split(r'\s|\t', line.strip()))
        if u < min_node_id:
            min_node_id = u
        if v < min_node_id:
            min_node_id = v
        edges.append((u, v))

if min_node_id != 0:
    LOG.warning("Nodes should start with index 0, but %d found. Fixing...", min_node_id)
    for i, (u, v) in enumerate(edges):
        edges[i] = (u - min_node_id, v - min_node_id)

    input_file = tmp_dir.joinpath("fixed_" + input_file.name)
    with input_file.open('w') as f:
        for u, v in edges:
            f.write("{} {}\n".format(u, v))


# Convert input file (list of edges) to a undirected list of edges
LOG.info("Converting input file to undirected list of edges...")
bidir_edges = []
for (u, v) in edges:
    bidir_edges.append((u, v))
    bidir_edges.append((v, u))

bidir_edges_file = tmp_dir / 'bidir_edges.txt'
with bidir_edges_file.open('w') as f:
    for u, v in bidir_edges:
        f.write("%s %s\n" % (u, v))

del edges
del bidir_edges

# Run pretraining
prefix = 'input-ds_'
pretrain_dir = tmp_dir / 'pretrain'
pretrain_dir.mkdir(parents=True, exist_ok=True)
LOG.info("Running pretraining...")
p = subprocess.Popen(
    [
        './magic',
        '-i', str(bidir_edges_file),
        '-o', str(pretrain_dir / prefix),
        '-nt', str(args.num_threads),
        '-c', str(args.n_communities),
        '-mc', str(args.min_communities),
        '-xc', str(args.max_communities),
        '-nc', str(args.n_trials),
        '-sa', str(args.alpha),
        '-sb', str(args.beta),
        '-st', str(args.allow_reference),
        '-woe', str(args.disable_eta),
        '-se', str(args.same_eta),
        '-mi', str(args.max_iter),
        '-si', str(args.save_iter),
        '-rsi', str(args.save_steps),
        '-sa', str(args.zero_threshold),
        '-lnf', str(args.remain_largest),
    ],
    shell=False,
    stdout=sys.stdout, stderr=sys.stderr,
    cwd='/app/src/PreTrain',
)
p.wait()
LOG.info("Pretraining finished.")

# Convert pretrain results
LOG.info("Converting PreTrain embeddings to CommunityGAN embeddings...")
input_embeddings_file = tmp_dir / 'input-ds_final.embeddings.txt'
p = subprocess.Popen(
    [
        'python', './scripts/format_transform.py',
        str(pretrain_dir / '{prefix}final.f.txt'.format(prefix=prefix)),
        str(input_embeddings_file),
    ],
    shell=False,
    stdout=sys.stdout, stderr=sys.stderr,
)
p.wait()

if args.ground_truth:
    LOG.info('Using provided ground truth communities.')
    ground_truth_file = pathlib.Path(args.ground_truth)
else:
    LOG.info("Converting community embeddings to CommunityGAN embeddings...")
    with pretrain_dir.joinpath('{prefix}final.cmty.txt'.format(prefix=prefix)).open('r') as f:
        communities = []
        for line in f.readlines():
            communities.append(
                [x[1:] for x in line.strip().split('\t') if x.strip() != '']
            )

    ground_truth_file = tmp_dir / 'ground_truth.txt'
    with ground_truth_file.open('w') as f:
        for community in communities:
            f.write('\t'.join(community) + '\n')

# Run CommunityGAN
result_filename = tmp_dir / 'result.txt'
emb_filenames_g_file = tmp_dir / '{prefix}_gen_.emb'.format(prefix=prefix)
emb_filenames_d_file = tmp_dir / '{prefix}_dis_.emb'.format(prefix=prefix)
LOG.info("Running CommunityGAN...")
p = subprocess.Popen(
    [
        'python', 'community_gan.py',
        'motif_size', str(args.motif_size),
        'batch_size_gen', str(args.batch_size_gen),
        'batch_size_dis', str(args.batch_size_dis),
        'n_sample_gen', str(args.n_sample_gen),
        'n_sample_dis', str(args.n_sample_dis),
        'lr_gen', str(args.lr_gen),
        'lr_dis', str(args.lr_dis),
        'n_epochs', str(args.n_epochs),
        'n_epochs_gen', str(args.n_epochs_gen),
        'n_epochs_dis', str(args.n_epochs_dis),
        'gen_interval', str(args.gen_interval),
        'dis_interval', str(args.dis_interval),
        'update_ratio', str(args.update_ratio),
        'max_value', str(args.max_value),
        'load_model', str(args.load_model),
        'save_steps', str(args.save_steps),
        'n_emb', str(args.n_emb),
        'num_threads', str(args.num_threads),
        'window_size', str(args.window_size),
        'app', str(args.app),
        'dataset', str(args.dataset),
        'train_filename', str(input_file),
        'pretrain_emb_filename_d', str(input_embeddings_file),
        'pretrain_emb_filename_g', str(input_embeddings_file),
        'community_filename', str(ground_truth_file),
        'log', str(output_dir / 'log'),
        'cache_filename_prefix', str(output_dir / 'cache' / prefix),
        'result_filename', str(result_filename),
        'emb_filenames_g', str(emb_filenames_g_file),
        'emb_filenames_d', str(emb_filenames_d_file),
    ],
    shell=False,
    stdout=sys.stdout, stderr=sys.stderr,
    cwd='/app/src/CommunityGAN'
)
p.wait()


LOG.info('Extracting communities from CommunityGAN embeddings...')
embeddings_file = emb_filenames_g_file
with embeddings_file.open('r') as f:
    n_node, n_embed = map(int, f.readline().strip().split('\t'))
    embedding_matrix = np.random.rand(n_node, n_embed)
    for line in f:
        emd = line.split('\t')
        embedding_matrix[int(emd[0]), :] = [float(item) for item in emd[1:]]

LOG.info('Writing non-overlapping communities to file...')
communities = defaultdict(list)
for i, ci in enumerate(embedding_matrix.argmax(axis=1)):
    communities[ci].append(i)

with (output_dir / 'default.coms').open('w') as f:
    for community in communities.values():
        f.write(' '.join(map(str, community)) + '\n')

LOG.info('Writing overlapping communities to file...')
communities_overlapping = defaultdict(list)
for i in range(n_node):
    cis = np.where(embedding_matrix[i, :] > args.overlap_threshold)[0]
    for ci in cis:
        communities_overlapping[ci].append(i)

with (output_dir / 'overlapping.coms').open('w') as f:
    for community in communities_overlapping.values():
        f.write(' '.join(map(str, community)) + '\n')
