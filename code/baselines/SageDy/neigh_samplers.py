"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division
from __future__ import print_function
import numpy as np

from layers import Layer

import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""
class WeightNeighborSampler(Layer):
    """
    Samples beighbors with weight 
    """
    def __init__(self, adj_info, graph, **kwargs):
        super(WeightNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.graph = graph
        new_adj_info = []
        for i in range(len(self.adj_info)):
            total_weights = []
            for j in self.adj_info[i]:
                try:
                    weight = self.graph[i][int(j)]['weight']
                except KeyError:
                    weight = 0
                    total_weights.append(weight)
                else:
                    total_weights.append(weight)

            large_to_small_neigh = []
            for k in range(len(total_weights)):
                max_index = total_weights.index(max(total_weights))
                max_neigh = self.adj_info[i][max_index]
                large_to_small_neigh.append(max_neigh)
                total_weights[max_index] = -1
            new_adj_info.append(large_to_small_neigh)
        self.new_adj_info = np.array(new_adj_info)
        self.new_adj_info = tf.constant(self.new_adj_info)
        self.new_adj_info = tf.cast(self.new_adj_info, dtype = tf.int32)
        
    
    def _call(self, inputs):
        ids, num_samples = inputs
               
        adj_lists = tf.nn.embedding_lookup(self.new_adj_info, ids)
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        
        return adj_lists


class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        
                
    def _call(self, inputs):
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)  
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])

        return adj_lists
