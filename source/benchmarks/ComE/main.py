__author__ = 'ando'

import argparse
import os
import random
from multiprocessing import cpu_count
import logging as log
from pathlib import Path

import networkx as nx
import numpy as np
import psutil
from math import floor
from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec
import utils.IO_utils as io_utils
from utils.IO_utils import exports_path, cache_path, outputs_path
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import timeit

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass


def load_graph(path):
    path = Path(path).absolute()
    print("Loading graph from {}".format(path))
    train_graph_path = path / "train.gpickle"
    G_train = nx.read_gpickle(train_graph_path)
    return G_train

def random_string():
    """Generate a random string."""
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(10))



if __name__ == "__main__":
    # Reading the input parameters form the configuration files
    parser = argparse.ArgumentParser(description='ComE')
    parser.add_argument('--dataset', type=str, default='StarWars')
    parser.add_argument('--dataset_version', type=str, default='base')
    parser.add_argument('--run_name', type=str, default=random_string())
    # parser.add_argument('--ground_truth', type=str, default='/input/ground_truth.txt', help='input ground truth co')
    parser.add_argument('--number_walks', type=int, default=10, help='number of walks')
    parser.add_argument('--walk_length', type=int, default=20, help='length of each walk')
    parser.add_argument('--representation_size', type=int, default=128, help='size of the embedding')
    parser.add_argument('--num_workers', type=int, default=10, help='number of thread')
    parser.add_argument('--num_iter', type=int, default=1, help='number of overall iteration')
    parser.add_argument('--reg_covar', type=float, default=0.00001,
                        help='regularization coefficient to ensure positive covar')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--window_size', type=int, default=10,
                        help='windows size used to compute the context embedding')
    parser.add_argument('--negative', type=int, default=5, help='number of negative sample')
    parser.add_argument('--lr', type=float, default=0.025, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value')
    parser.add_argument('--beta', type=float, default=0.1, help='beta value')
    parser.add_argument('--down_sampling', type=float, default=0.0, help='down sampling rate')
    parser.add_argument('--ks', type=int, default=5, help='k values')
    parser.add_argument('--table_size', type=int, default=1000000, help='Table size for the node2vec')
    parser.add_argument('--k', type=int, default=5, help='Amount of communities to be found')
    args = parser.parse_args()

    cache_dir = cache_path / args.dataset / args.dataset_version
    dataset_path = exports_path / args.dataset / args.dataset_version
    outputs_dir = outputs_path / 'ComE' / args.dataset / args.dataset_version / args.run_name

    # CONSTRUCT THE GRAPH
    G = load_graph(dataset_path)

    # Sampling the random walks for context
    log.info("sampling the paths")

    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        walk_files = graph_utils.write_walks_to_disk(
            G, cache_dir / f'walks',
            num_paths=args.number_walks,
            path_length=args.walk_length,
            alpha=0,
            rand=random.Random(0),
            num_workers=args.num_workers
        )
    else:
        log.info("Loading the walks from the cache")
        walk_files = list(cache_dir.glob('walks.*'))

    pretrain_file = cache_dir / 'pretrain.bin'
    context_total_path = G.number_of_nodes() * args.number_walks * args.walk_length
    vertex_counts = graph_utils.count_textfiles(walk_files, args.num_workers)
    model = Model(
        vertex_counts,
        size=args.representation_size,
        down_sampling=args.down_sampling,
        table_size=args.table_size,
        k=args.k
    )

    # Learning algorithm
    node_learner = Node2Vec(workers=args.num_workers, negative=args.negative, lr=args.lr)
    cont_learner = Context2Vec(window_size=args.window_size, workers=args.num_workers, negative=args.negative,
        lr=args.lr)
    com_learner = Community2Vec(lr=args.lr)
    edges = np.array(G.edges())

    if not pretrain_file.exists():
        log.debug("context_total_path: %d" % (context_total_path))
        log.debug('node total edges: %d' % G.number_of_edges())

        log.info('\n_______________________________________')
        log.info('\t\tPRE-TRAINING\n')
        ###########################
        #   PRE-TRAINING          #
        ###########################
        node_learner.train(model,
            edges=edges,
            iter=1,
            chunksize=args.batch_size)

        cont_learner.train(model,
            paths=graph_utils.combine_files_iter(walk_files),
            total_nodes=context_total_path,
            alpha=1,
            chunksize=args.batch_size)
        #
        model.save(pretrain_file)
    else:
        log.info("Using the pretrained model")

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    iter_node = floor(context_total_path / G.number_of_edges())
    iter_com = floor(context_total_path / (G.number_of_edges()))
    # iter_com = 1
    # alpha, beta = alpha_betas

    log.info('\n_______________________________________\n')
    model = model.load_model(pretrain_file)
    model.reset_communities_weights(args.k)
    log.info('using alpha:{}\tbeta:{}\titer_com:{}\titer_node: {}'.format(args.alpha, args.beta, iter_com, iter_node))
    start_time = timeit.default_timer()

    com_learner.fit(model, reg_covar=args.reg_covar, n_init=10)
    node_learner.train(model,
        edges=edges,
        iter=iter_node,
        chunksize=args.batch_size)

    com_learner.train(G.nodes(), model, args.beta, chunksize=args.batch_size, iter=iter_com)

    cont_learner.train(model,
        paths=graph_utils.combine_files_iter(walk_files),
        total_nodes=context_total_path,
        alpha=args.alpha,
        chunksize=args.batch_size)

    log.info('time: %.2fs' % (timeit.default_timer() - start_time))
    Z = model.node_embedding
    mus = model.centroid
    z = model.pi.argmax(axis=1)

    outputs_dir.mkdir(exist_ok=True, parents=True)
    np.save(outputs_dir / 'assignments.npy', z)
    np.save(outputs_dir / 'embeddings.npy', Z)
    np.save(outputs_dir / 'means.npy', mus)

    # log.info(model.centroid)
    # io_utils.save_embedding(model.node_embedding, model.vocab,
    #     file_name="{}_alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_ds-{}".format(output_file,
    #         args.alpha,
    #         args.beta,
    #         args.window_size,
    #         args.negative,
    #         args.lr,
    #         iter_com,
    #         iter_node,
    #         model.k,
    #         args.down_sampling)
    # )
