import unittest
import os

import faiss
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tf_euclidean_distmat(xs, agg_fn: tf.reduce_min, inf: tf.Tensor = None):
    n = xs.shape[0]
    mat = tf.repeat(tf.expand_dims(xs, axis=0), repeats=n, axis=0)
    diff = tf.subtract(mat, tf.expand_dims(xs, axis=1))
    if inf is None:
        inf = tf.constant(np.PINF) if agg_fn == tf.reduce_min else tf.constant(np.NINF)

    dist_mat = tf.reduce_sum(tf.square(diff), axis=-1)
    dist = tf.where(tf.cast(tf.eye(n, n), dtype=tf.bool), inf, dist_mat)
    dist_agg = tf.sqrt(agg_fn(dist, axis=-1))
    return dist_agg


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        tf.random.set_seed(42)
        self.n_clusters = 3
        self.n_nodes = 20
        self.dimensions = 2

        self.clusters = tf.random.uniform(shape=(self.n_clusters, self.dimensions)) * 10
        self.nn_index = faiss.IndexFlatL2(self.dimensions)
        self.nn_index.add(self.clusters.numpy())

        self.nodes = tf.random.normal(shape=(self.n_nodes, self.dimensions)).numpy()
        self.cluster_nodes = [3, 5, 12]
        for cluster, (offset, size) in enumerate(zip(np.cumsum([0, *self.cluster_nodes[:-1]]), self.cluster_nodes)):
            self.nodes[offset:offset + size] = self.nodes[offset:offset + size] + self.clusters[cluster]
        self.nodes = tf.convert_to_tensor(self.nodes)

        self.membership = tf.squeeze(tf.convert_to_tensor(
            self.nn_index.assign(self.nodes.numpy(), k=1),
            dtype=tf.int32
        ))

    def test_cluster_separation(self):
        tf.reduce_mean(tf_euclidean_distmat(self.clusters, agg_fn=tf.reduce_min))
        u = 0

    def test_davies_bouldin_index(self):
        # Calculate S(i)
        node_clusters = tf.gather(self.clusters, tf.squeeze(self.membership))
        diff = tf.reduce_sum(tf.square(tf.subtract(self.nodes, node_clusters)), axis=-1)
        cluster_mass = tf.sqrt(tf.math.segment_sum(diff, self.membership))
        cluster_counts = tf.math.segment_sum(tf.ones_like(self.membership), self.membership)
        values = cluster_mass / tf.cast(cluster_counts, dtype=tf.float32)

        # Calculate Rij
        mat = tf.repeat(tf.expand_dims(self.clusters, axis=0), repeats=self.n_clusters, axis=0)
        diff = tf.subtract(mat, tf.expand_dims(self.clusters, axis=1))
        bot = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))

        mat = tf.repeat(tf.expand_dims(values, axis=0), repeats=self.n_clusters, axis=0)
        top = tf.add(mat, tf.expand_dims(values, axis=1))

        rij_raw = tf.divide(top, bot)
        rij = tf.where(tf.cast(tf.eye(self.n_clusters), dtype=tf.bool), tf.constant(np.NINF, dtype=tf.float32), rij_raw)
        Di = tf.reduce_max(rij, axis=-1)

        u = 0
