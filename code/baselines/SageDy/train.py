from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import networkx as nx
from collections import Counter
import logging
import scipy

from link_prediction import evaluate_classifier, write_to_csv
from flags import *
from models import SageDy
import minibatch as um
import preprocess as up
import utilities as uu 

np.random.seed(123)
tf.set_random_seed(123)

flags = tf.app.flags
FLAGS = flags.FLAGS
output_dir = "./logs/{}_{}/".format(FLAGS.base_model, FLAGS.model)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

config_file = output_dir + "flags_{}.json".format(FLAGS.dataset)

print("Updated flags", FLAGS.flag_values_dict().items())

LOG_DIR = output_dir + FLAGS.log_dir
SAVE_DIR = output_dir + FLAGS.save_dir
CSV_DIR = output_dir + FLAGS.csv_dir
MODEL_DIR = output_dir + FLAGS.model_dir

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.isdir(CSV_DIR):
    os.mkdir(CSV_DIR)

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()

log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.info(FLAGS.flag_values_dict().items())

output_file = CSV_DIR + '/%s_%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

num_time_steps = FLAGS.time_steps
identity_dim = FLAGS.identity_dim

graphs, adjs = up.load_graphs(FLAGS.dataset)
if FLAGS.featureless:
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
else:
    feats = up.load_feats(FLAGS.dataset)
num_features = feats[0].shape[1]
assert num_time_steps < len(adjs) + 1  # So that, (t+1) can be predicted.

adj_train = []
feats_train = []
num_features_nonzero = []
loaded_pairs = False

context_pairs_train = up.get_context_pairs(graphs, num_time_steps)

train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    up.get_evaluation_data(adjs, num_time_steps, FLAGS.dataset)

new_G = nx.MultiGraph()
new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

for e in graphs[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])

graphs[num_time_steps - 1] = new_G
adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)

for i in range(num_time_steps-1):
    new_G = nx.MultiGraph()
    new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))
    for e in graphs[i].edges():
        new_G.add_edge(e[0], e[1])
    graphs[i] = new_G
    adjs[i] = nx.adjacency_matrix(graphs[i])
    
for graph in graphs:
    res = Counter(list(graph.edges()))
    for edge in graph.edges():
        weight = res[edge]
        a = edge[0]
        b = edge[1]
        graph[a][b].update({'weight':weight})

print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))
logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

adj_train = list(map(lambda adj: up.normalize_graph_gcn(adj), adjs))

if FLAGS.featureless:  
    feats = [scipy.sparse.rand(adjs[num_time_steps - 1].shape[0], identity_dim).tocsr()[range(0, x.shape[0]), :] for x in feats if
             x.shape[0] <= feats[num_time_steps - 1].shape[0]]
    model_feats = [x.todense() for x in feats]
num_features = feats[0].shape[1]

feats_train = list(map(lambda feat: up.preprocess_features(feat)[1], feats))
num_features_nonzero = [x[1].shape[0] for x in feats_train]

def construct_placeholders(num_time_steps):
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(num_time_steps - FLAGS.window - 1, 0)
    placeholders = {
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),  # [None,1]
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, num_time_steps)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop'),
        'dropout':tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':tf.placeholder(tf.int32)
    }
    return placeholders


print("Initializing session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

placeholders = construct_placeholders(num_time_steps)

minibatchIterator = um.NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          placeholders, num_time_steps, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train)
print("# training batches per epoch", minibatchIterator.num_training_batches())

model = SageDy(placeholders, minibatchIterator.degs, graphs, adjs, model_feats)
total_minibatch = model.total_minibatch
total_adj_info_ph = model.total_adj_info_ph
feed_dict = {}
for i in range(FLAGS.time_steps):
    feed_dict[total_adj_info_ph[i]] = total_minibatch[i].adj
sess.run(tf.global_variables_initializer(), feed_dict = feed_dict)

epochs_test_result = uu.defaultdict(lambda: [])
epochs_val_result = uu.defaultdict(lambda: [])
epochs_embeds = []
epochs_attn_wts_all = []

for epoch in range(FLAGS.epochs):
    minibatchIterator.test_reset()
    minibatchIterator.shuffle()
    epoch_loss = 0.0
    it = 0
    print('Epoch: %04d' % (epoch + 1))
    epoch_time = 0.0
    while not minibatchIterator.end():
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        t = time.time()
        
        _, train_cost, graph_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],
                                                       feed_dict=feed_dict)
        epoch_time += time.time() - t
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, train_cost))
        logging.info("Time for Mini batch : {}".format(time.time() - t))

        epoch_loss += train_cost
        it += 1

    print("Time for epoch ", epoch_time)
    logging.info("Time for epoch : {}".format(epoch_time))
    if (epoch + 1) % FLAGS.test_freq == 0:
        minibatchIterator.test_reset()
        embs = []
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        if FLAGS.window < 0:
            assert FLAGS.time_steps == model.final_output_embeddings.get_shape()[1]
        while not minibatchIterator.end():
            feed_dict = minibatchIterator.next_minibatch_feed_dict()
            emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:,
                          model.final_output_embeddings.get_shape()[1] - 2, :]
            emb = np.array(emb)
            embs.append(emb)
        total_embs = embs[0] 
        for i in range(len(embs)-1):
            total_embs = np.vstack((total_embs, embs[i+1]))
        
        embs = total_embs

        
        try:
            val_results, test_results, _, _ = evaluate_classifier(train_edges,
                                                                  train_edges_false, val_edges, val_edges_false, test_edges,
                                                                  test_edges_false, embs, embs)
        except ValueError:
            break
        else:
            epoch_auc_val = val_results["HAD"][1]
            epoch_auc_test = test_results["HAD"][1]
    
            print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
            print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
            logging.info("Val results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_val))
            logging.info("Test results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_test))
    
            epochs_test_result["HAD"].append(epoch_auc_test)
            epochs_val_result["HAD"].append(epoch_auc_val)
            epochs_embeds.append(embs)
    epoch_loss /= it
    print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"]))

print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))

val_results, test_results, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges, val_edges_false,
                                                      test_edges, test_edges_false, epochs_embeds[best_epoch],
                                                      epochs_embeds[best_epoch])

print("Best epoch val results {}\n".format(val_results))
print("Best epoch test results {}\n".format(test_results))

logging.info("Best epoch val results {}\n".format(val_results))
logging.info("Best epoch test results {}\n".format(test_results))

write_to_csv(val_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='val')
write_to_csv(test_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='test')

emb = epochs_embeds[best_epoch]
np.savez(SAVE_DIR + '/{}_embs_{}_{}.npz'.format(FLAGS.model, FLAGS.dataset, FLAGS.time_steps - 2), data=emb)
sess.close()
print('完成')