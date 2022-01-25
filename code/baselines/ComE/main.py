__author__ = 'ando'

import argparse
import logging as log
import os
import pathlib
import random
import timeit
from collections import defaultdict
from math import floor
from multiprocessing import cpu_count

import networkx as nx
import numpy as np
import psutil
from sklearn.neighbors import NearestNeighbors

import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
from ADSCModel.community_embeddings import Community2Vec
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.model import Model
from ADSCModel.node_embeddings import Node2Vec

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass


def save_communities(model, output_file):
    centroids = model.centroid
    embeddings = model.node_embedding
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centroids)
    distances, indices = nn.kneighbors(embeddings)

    vocab = {}
    if isinstance(vocab, dict):
        for i, node in model.vocab.items():
            vocab[node.index] = i
    else:
        for i in range(embeddings.shape[0]):
            vocab[i] = i

    comlist = []
    for i, index in enumerate(indices):
        comlist.append((vocab[i], index[0]))

    with open(output_file, 'w') as f:
        for nid, cid in sorted(comlist):
            f.write('{}\t{}\n'.format(nid, cid))


if __name__ == "__main__":
    # Reading the input parameters form the configuration files
    parser = argparse.ArgumentParser(description='ComE')
    parser.add_argument('--input', type=str, default='/input', help='input directory to search for input file')
    parser.add_argument('--output', type=str, default='/output', help='output directory to save the output file')
    # parser.add_argument('--ground_truth', type=str, default='/input/ground_truth.txt', help='input ground truth co')
    parser.add_argument('--number_walks', type=int, default=10, help='number of walks')
    parser.add_argument('--walk_length', type=int, default=80, help='length of each walk')
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
    parser.add_argument('--table_size', type=int, default=100000000, help='Table size for the node2vec')
    parser.add_argument('--k', type=int, default=5, help='Amount of communities to be found')
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    tmp_dir = output_dir.joinpath('tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    number_walks = args.number_walks
    walk_length = args.walk_length
    representation_size = args.representation_size
    num_workers = args.num_workers
    num_iter = args.num_iter
    reg_covar = args.reg_covar
    batch_size = args.batch_size
    window_size = args.window_size
    negative = args.negative
    lr = args.lr
    alpha_betas = [(args.alpha, args.beta)]
    down_sampling = args.down_sampling
    ks = [args.ks]
    k = args.k

    walks_filebase = tmp_dir.joinpath('randomwalks')  # where read/write the sampled path
    walks_filebase.mkdir(parents=True, exist_ok=True)

    # CONSTRUCT THE GRAPH
    input_file = str(next(input_dir.glob('*.edgelist')))
    with open(input_file, 'rb') as f:
        G = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=int)
        G.to_undirected()

    # G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    # Sampling the random walks for context
    log.info("sampling the paths")
    walk_files = graph_utils.write_walks_to_disk(G, str(walks_filebase.joinpath('walks')),
                                                 num_paths=number_walks,
                                                 path_length=walk_length,
                                                 alpha=0,
                                                 rand=random.Random(0),
                                                 num_workers=num_workers)

    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)
    model = Model(vertex_counts,
                  size=representation_size,
                  down_sampling=down_sampling,
                  table_size=args.table_size,
                  # path_labels=args.ground_truth,
                  path_labels='none',
                  k=k
                  )

    # Learning algorithm
    node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
    com_learner = Community2Vec(lr=lr)

    context_total_path = G.number_of_nodes() * number_walks * walk_length
    edges = np.array(G.edges())
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
                       chunksize=batch_size)

    cont_learner.train(model,
                       paths=graph_utils.combine_files_iter(walk_files),
                       total_nodes=context_total_path,
                       alpha=1,
                       chunksize=batch_size)
    #
    model.save(str(tmp_dir.joinpath('pre-training')))


    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    iter_node = floor(context_total_path / G.number_of_edges())
    iter_com = floor(context_total_path / (G.number_of_edges()))
    # iter_com = 1
    # alpha, beta = alpha_betas

    for it in range(num_iter):
        for alpha, beta in alpha_betas:
            for k in ks:
                output_file_path = tmp_dir.joinpath(
                    "alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_ds-{}".format(
                        alpha,
                        beta,
                        window_size,
                        negative,
                        lr,
                        iter_com,
                        iter_node,
                        model.k,
                        down_sampling)
                )
                output_file_path.mkdir(parents=True, exist_ok=True)
                log.info('\n_______________________________________\n')
                log.info('\t\tITER-{}\n'.format(it))
                model = model.load_model(str(tmp_dir.joinpath('pre-training')))
                model.reset_communities_weights(k)
                log.info('using alpha:{}\tbeta:{}\titer_com:{}\titer_node: {}'.format(alpha, beta, iter_com, iter_node))
                start_time = timeit.default_timer()

                com_learner.fit(model, reg_covar=reg_covar, n_init=10)
                node_learner.train(model,
                                   edges=edges,
                                   iter=iter_node,
                                   chunksize=batch_size)

                com_learner.train(G.nodes(), model, beta, chunksize=batch_size, iter=iter_com)

                cont_learner.train(model,
                                   paths=graph_utils.combine_files_iter(walk_files),
                                   total_nodes=context_total_path,
                                   alpha=alpha,
                                   chunksize=batch_size)

                model.save(str(output_file_path.joinpath('model_trained')))
                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
                io_utils.save_embedding(model.node_embedding, model.vocab,
                                        file_name=str(output_file_path.joinpath('node_embedding')))

                io_utils.save_embedding(model.centroid, list(range(len(model.centroid))),
                                        file_name=str(output_file_path.joinpath('community_centroids')))
                save_communities(model, output_file_path.joinpath('static.comlist'))
                save_communities(model, output_dir.joinpath('static.comlist'))
