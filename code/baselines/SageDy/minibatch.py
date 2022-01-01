#导入python未来支持的语言特征division(精确除法)
from __future__ import division
#即使在python2.X，使用print就得像python3.X那样加括号使用。
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.adj, self.deg = self.construct_adj()

    def construct_adj(self):
        adj = np.ones((len(self.id2idx), self.max_degree))
        row = np.ones(self.max_degree)
        for i in range(len(self.id2idx)):
            adj[self.id2idx[i], :] = i * row
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.nodes:
            neighbors = np.array([self.id2idx[neighbor] for neighbor in self.G.neighbors(nodeid)])              
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes to sample context pairs for a batch of nodes.

    graphs -- list of networkx graphs
    features -- list of (scipy) sparse node attribute matrices
    adjs -- list of adj matrices (of the graphs)
    placeholders -- standard tensorflow placeholders object for feeding
    num_time_steps -- number of graphs to train +1
    context_pairs -- list of (target, context) pairs obtained from random walk sampling.
    batch_size -- size of the minibatches (# nodes)
    """
    def __init__(self, graphs, features, adjs, placeholders, num_time_steps, context_pairs=None, batch_size=100):

        self.graphs = graphs
        self.features = features
        self.adjs = adjs
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.batch_num = 0
        self.num_time_steps = num_time_steps
        self.degs = self.construct_degs()
        self.context_pairs = context_pairs
        self.max_positive = FLAGS.neg_sample_size
        self.train_nodes = self.graphs[num_time_steps-1].nodes() # all nodes in the graph.
        print ("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        degs = []
        for i in range(0, self.num_time_steps):
            G = self.graphs[i]
            deg = np.zeros((len(G.nodes())))
            for nodeid in G.nodes():
                neighbors = np.array(list(G.neighbors(nodeid)))
                deg[nodeid] = len(neighbors)
            degs.append(deg)
        min_t = 0
        if FLAGS.window > 0:
            min_t = max(self.num_time_steps - FLAGS.window - 1, 0)
        return degs[min_t:]

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes):
        """ Feed dict with (a) node pairs, (b) list of attribute matrices (c) list of snapshot adjs and metadata"""
        node_1_all = []
        node_2_all = []
        min_t = 0
        if FLAGS.window > 0:
            min_t = max(self.num_time_steps - FLAGS.window - 1, 0)
        for t in range(min_t, self.num_time_steps):
            node_1 = []
            node_2 = []

            for n in batch_nodes:
                if len(self.context_pairs[t][n]) > self.max_positive:
                    node_1.extend([n]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][n], self.max_positive, replace=False))
                else:
                    node_1.extend([n]* len(self.context_pairs[t][n]))
                    node_2.extend(self.context_pairs[t][n])

            assert len(node_1) == len(node_2)
            assert len(node_1) <= self.batch_size * self.max_positive

            feed_dict = dict()
            node_1_all.append(node_1)
            node_2_all.append(node_2)

        feed_dict.update({self.placeholders['node_1'][t-min_t]:node_1_all[t-min_t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['node_2'][t-min_t]:node_2_all[t-min_t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['features'][t-min_t]:self.features[t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['adjs'][t-min_t]: self.adjs[t] for t in range(min_t, self.num_time_steps)})

        feed_dict.update({self.placeholders['batch_nodes']:np.array(batch_nodes).astype(np.int32)})
        return feed_dict

    def num_training_batches(self):
        """ Compute the number of training batches (using batch size)"""
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        """ Return the feed_dict for the next minibatch (in the current epoch) with random shuffling"""
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        """ Reset batch number"""
        self.train_nodes =  self.graphs[self.num_time_steps-1].nodes()
        self.batch_num = 0