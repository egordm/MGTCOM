import layers as la
import tensorflow as tf
from collections import namedtuple
from neigh_samplers import UniformNeighborSampler, WeightNeighborSampler
import minibatch as um
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
               
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        
        #Build sequential layer model
        self.activations.append(self.inputs)
        
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        
        self.outputs = self.activations[-1]     
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])
    
class SageDy(Model):
    def _accuracy(self):
        pass

    def __init__(self, placeholders, degrees, graphs, adjs, features,  
                  **kwargs):
        super(SageDy, self).__init__(**kwargs)
        self.features = features
        self.graphs = graphs
        self.adjs = adjs
        self.attn_wts_all = []
        self.temporal_layers = []
        self.temporal_attention_layers = []
        self.placeholders = placeholders
        self.temporal_head_config = list(map(int, FLAGS.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, FLAGS.temporal_layer_config.split(",")))
        if FLAGS.window < 0:
            self.num_time_steps = len(placeholders['features'])
        else:
            self.num_time_steps = min(len(placeholders['features']), FLAGS.window + 1)  # window = 0 => only self.
        self.num_time_steps_train = self.num_time_steps - 1
        self.degrees = degrees
        self.batch_size = placeholders['batch_nodes']
        self.inputs1 = placeholders['batch_nodes']
        id_maps = []      
        for i in range(FLAGS.time_steps): 
            id_map = {}
            for n in range(len(graphs[i].nodes())):
                id_map[n] = n
            id_maps.append(id_map)
        total_layer_infos = []
        total_minibatch = []
        total_adj_info_ph = []
        for i in range(FLAGS.time_steps): 
            minibatch = um.EdgeMinibatchIterator(graphs[i],
                                                 id_maps[i],
                                                 placeholders, batch_size=FLAGS.batch_size,
                                                 max_degree=FLAGS.max_degree)
            total_minibatch.append(minibatch)
            adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
            total_adj_info_ph.append(adj_info_ph)
            adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
            sampler1 = WeightNeighborSampler(minibatch.adj, graphs[i])
            sampler2 = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler1, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler2, FLAGS.samples_2, FLAGS.dim_2)]
            total_layer_infos.append(layer_infos)
        self.layer_infos = total_layer_infos
        self.total_minibatch = total_minibatch
        self.total_adj_info_ph = total_adj_info_ph
        self.dims = [128, 128, 128]
        
        self._build()
        
    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            next_hidden = []
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [tf.shape(batch_size)[0] * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]                
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators
    
    def sample(self, inputs, layer_infos, batch_size=None):
        if batch_size is None:
            batch_size = tf.shape(self.batch_size)[0]
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def _build(self):
        proximity_labels = [tf.expand_dims(tf.cast(self.placeholders['node_2'][t], tf.int64), 1)
                            for t in range(0, len(self.placeholders['features']))]  # [B, 1]

        self.proximity_neg_samples = []
        for t in range(len(self.placeholders['features']) - 1 - self.num_time_steps_train,
                       len(self.placeholders['features']) - 1):
            self.proximity_neg_samples.append(tf.nn.fixed_unigram_candidate_sampler(
                true_classes=proximity_labels[t],
                num_true=1,
                num_sampled=FLAGS.neg_sample_size,
                unique=False,
                range_max=len(self.degrees[t]),
                distortion=0.75,
                unigrams=self.degrees[t].tolist())[0])

        # Build actual model.
        self.final_output_embeddings = self.build_net(self.temporal_head_config, self.temporal_layer_config, self.placeholders['temporal_drop'])
        self._loss()
        self.init_optimizer()

    def build_net(self, temporal_head_config, temporal_layer_config, temporal_drop):
        
        # 1: Structural aggregate Layers(GCN) 
        if FLAGS.aggregator_layer == 'graphsage_mean':
            self.aggregator_cls = la.MeanAggregator
        elif FLAGS.aggregator_layer == 'graphsage_seq':
            self.aggregator_cls = la.SeqAggregator
        elif FLAGS.aggregator_layer == 'graphsage_maxpool':
            self.aggregator_cls = la.MaxPoolingAggregator
        elif FLAGS.aggregator_layer == 'graphsage_meanpool':
            self.aggregator_cls = la.MeanPoolingAggregator
        elif FLAGS.aggregator_layer == "gcn":
            self.aggregator_cls = la.GCNAggregator
        elif FLAGS.aggregator_layer == 'sagedy':
            self.aggregator_cls = la.SagedyAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
        
        # 2: Temporal Attention Layers
        input_dim = 128
        for i in range(0, len(temporal_layer_config)):
            if i > 0:
                input_dim = temporal_layer_config[i - 1]
            temporal_layer = la.TemporalAttentionLayer(input_dim=input_dim, n_heads=temporal_head_config[i],
                                                    attn_drop=temporal_drop, num_time_steps=self.num_time_steps,
                                                    residual=False)
            self.temporal_attention_layers.append(temporal_layer)        
        # 3: Structural Attention forward     
        total_samples1 = []
        total_support_sizes1 = []
        total_num_samples = []
        self.total_outputs1 = []
        self.total_aggregators = []
        for i in range(FLAGS.time_steps):
            samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos[i])
            total_samples1.append(samples1)
            total_support_sizes1.append(support_sizes1)   
            num_samples = [layer_info.num_samples for layer_info in self.layer_infos[i]]
            total_num_samples.append(num_samples)
            self.outputs1, self.aggregators = self.aggregate(total_samples1[i], self.features[i], self.dims, total_num_samples[i],
                total_support_sizes1[i], concat=False, model_size='small')
            self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
            self.outputs1 = tf.expand_dims(self.outputs1,axis=0)
            self.total_outputs1.append(self.outputs1)
            self.total_aggregators.append(self.aggregators)
        for t in range(0, self.num_time_steps):
            zero_padding = tf.zeros(
                [1, tf.shape(self.total_outputs1[-1])[1] - tf.shape(self.total_outputs1[t])[1], FLAGS.hidden1])
            self.total_outputs1[t] = tf.concat([self.total_outputs1[t], zero_padding], axis=1)

        structural_outputs = tf.transpose(tf.concat(self.total_outputs1, axis=0), [1, 0, 2])  # [N, T, F]
        noinfluence_structural_outputs = tf.reshape(structural_outputs,
                                        [-1, self.num_time_steps, FLAGS.hidden1])  # [N, T, F]
        
        #adding extra first influence 
        first_influence_outputs = []
        for time in range(self.num_time_steps):
            time_spacing = self.num_time_steps - 1 - time 
            delta = tf.get_variable('delta'+str(time), shape=[], initializer=tf.constant_initializer(1))
            time_spacing_intensity = tf.multiply(delta, time_spacing)
            influence_factor1 = tf.exp(-time_spacing_intensity)
            node_embeddings = noinfluence_structural_outputs[:, time, :]
            node_embeddings = tf.divide(node_embeddings, 1/influence_factor1)
            first_influence_outputs.append(node_embeddings)
        first_influence_outputs = tf.transpose(first_influence_outputs, [1, 0, 2])
        structural_outputs = tf.reshape(first_influence_outputs,
                                        [-1, self.num_time_steps, FLAGS.hidden1])
        
        #adding extra second influence
        information_change_total = []
        for time in range(self.num_time_steps - 1):
            information_change = tf.subtract(noinfluence_structural_outputs[:, time, :],
                                             noinfluence_structural_outputs[:, time + 1, :])
            information_change = tf.abs(information_change)
            information_change = tf.reduce_sum(information_change)
            information_change_total.append(information_change)
        max_index = tf.arg_max(information_change_total, 0)
        history_time = max_index + 1
        delta = tf.get_variable('delta', shape=[], initializer=tf.constant_initializer(1))
        current_time = self.num_time_steps - 1
        time_spacing = current_time - history_time
        node_embedding = noinfluence_structural_outputs[:, history_time, :]
        time_spacing = tf.cast(time_spacing, dtype = tf.float32)
        time_spacing_intensity = tf.multiply(delta, time_spacing)
        influence_factor2 = tf.exp(-time_spacing_intensity) + 1
        influence_structure_output = tf.divide(node_embedding, 1/influence_factor2)
        influence_structure_output = tf.add(influence_structure_output, structural_outputs[:,history_time,:])
        influence_structure_output = tf.divide(influence_structure_output, 2)
        second_influence_outputs = []
        for time in range(self.num_time_steps):
            tensor_time = tf.constant(time)
            if tensor_time == history_time:
                second_influence_outputs.append(influence_structure_output)
            else:
                node_embeddings = structural_outputs[:, time, :]
                second_influence_outputs.append(node_embeddings)
        second_influence_outputs = tf.transpose(second_influence_outputs, [1, 0, 2])
        structural_outputs = tf.reshape(second_influence_outputs,
                                        [-1, self.num_time_steps, FLAGS.hidden1])
        # 5: Temporal Attention forward
        temporal_inputs = structural_outputs
        for temporal_layer in self.temporal_attention_layers:
            outputs = temporal_layer(temporal_inputs)  # [N, T, F]
            temporal_inputs = outputs
            self.attn_wts_all.append(temporal_layer.attn_wts_all)
        return outputs

    def _loss(self):
        self.graph_loss = tf.constant(0.0)
        num_time_steps_train = self.num_time_steps_train
        for t in range(self.num_time_steps_train - num_time_steps_train, self.num_time_steps_train):
            output_embeds_t = tf.nn.embedding_lookup(tf.transpose(self.final_output_embeddings, [1, 0, 2]), t)
            inputs1 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_1'][t])
            inputs2 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_2'][t])
            pos_score = tf.reduce_sum(inputs1 * inputs2, axis=1)
            neg_samples = tf.nn.embedding_lookup(output_embeds_t, self.proximity_neg_samples[t])
            neg_score = (-1.0) * tf.matmul(inputs1, tf.transpose(neg_samples))
            pos_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score)
            neg_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(neg_score), logits=neg_score)
            self.graph_loss += tf.reduce_mean(pos_ent) + FLAGS.neg_weight * tf.reduce_mean(neg_ent)

        self.reg_loss = tf.constant(0.0)
        if len([v for v in tf.trainable_variables() if "neigh_weights" in v.name and 'self_weights' in v.name and "bias" not in v.name]) > 0:
            self.reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                       if "neigh_weights" in v.name and 'self_weights' in v.name and "bias" not in v.name]) * FLAGS.weight_decay
        self.loss = self.graph_loss + self.reg_loss

    def init_optimizer(self):
        trainable_params = tf.trainable_variables()
        actual_loss = self.loss
        gradients = tf.gradients(actual_loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # Set the model optimization op.
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))
