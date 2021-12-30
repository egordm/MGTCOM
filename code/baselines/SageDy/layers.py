import tensorflow as tf
from inits import glorot, zeros

conv1d = tf.layers.conv1d

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors"""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)        
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype = tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out*(1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul(sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
    def _call(self, inputs):
        return inputs
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class LSTMRNN(Layer):
    def __init__(self, batch_size, **kwargs):
        super(LSTMRNN, self).__init__(**kwargs)
        self.n_steps = 3
        self.input_size = 128
        self.output_size = 128
        self.cell_size = 128
        self.num_time_steps = 3
        self.batch_size = tf.shape(batch_size)[0]
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.vars['position_embeddings'] = tf.get_variable('position_embeddings',
                                                                   dtype=tf.float32,
                                                                   shape=[self.num_time_steps, self.input_size],
                                                                   initializer=xavier_init)  # [T, F]      
        self.weights = {
                'in':tf.Variable(tf.random_normal([self.input_size, self.cell_size])),
                'out':tf.Variable(tf.random_normal([self.cell_size, self.output_size]))
                }
        self.biases = {
                'in':tf.Variable(tf.constant(0.1, shape = [self.cell_size])),
                'out':tf.Variable(tf.constant(0.1, shape = [self.output_size]))                
                }
    def _call(self, inputs):
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.num_time_steps), 0), [tf.shape(inputs)[0], 1])
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embeddings'], position_inputs)# [N, T, F]
        X = tf.reshape(temporal_inputs, [-1, self.input_size])
        X_in = tf.matmul(X, self.weights['in']) + self.biases['in']
        X_in = tf.reshape(X_in,[-1, self.n_steps, self.cell_size])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias = 1.0, state_is_tuple = True)
        _init_state = lstm_cell.zero_state(self.batch_size,dtype=tf.float32)                                      
        outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state = _init_state,time_major = False)
        activation_output = outputs
        activation_output = tf.reshape(activation_output, [-1, self.cell_size])
        results = tf.matmul(activation_output,self.weights['out']) + self.biases['out']
        results = tf.reshape(results, [self.batch_size, self.n_steps, self.output_size])
        return results

class Dense(Layer):
    """Dense layer.（全连接层）"""
    def __init__(self, input_dim, output_dim, dropout=0., 
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False, 
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.dropout(x, 1-self.dropout)

        output = tf.matmul(x, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    
class SagedyAggregator(Layer):
    
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(SagedyAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, neigh_input_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, input_dim], name='self_weights')
            self.vars['weights'] = glorot([input_dim, output_dim], 
                                                        name='weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def _call(self, inputs): 
        self_vecs, neigh_vecs = inputs
        sample_num = neigh_vecs.shape[1]

        neigh_vecs = tf.cast(neigh_vecs, dtype=tf.float32)
        self_vecs = tf.cast(self_vecs, dtype=tf.float32)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        
        neigh_vecs = tf.reshape(neigh_vecs, [-1, 128])
        from_neighs = tf.matmul(neigh_vecs, self.vars['neigh_weights'])
        from_neighs = tf.reshape(from_neighs, [-1, sample_num, 128])

        from_self = tf.matmul(self_vecs, self.vars['self_weights'])
        
        means = tf.reduce_mean(tf.concat([from_neighs, 
            tf.expand_dims(from_self, axis=1)], axis=1), axis=1)
        
        output = tf.matmul(means, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)
    
class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        


        neigh_vecs = tf.cast(neigh_vecs, dtype=tf.float32)
        self_vecs = tf.cast(self_vecs, dtype=tf.float32)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1) #在concat后其维数变为之前的2倍

        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)




class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        
        neigh_vecs = tf.cast(neigh_vecs, dtype=tf.float32)
        self_vecs = tf.cast(self_vecs, dtype=tf.float32)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        output = tf.matmul(means, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        
        neigh_vecs = tf.cast(neigh_vecs, dtype=tf.float32)
        self_vecs = tf.cast(self_vecs, dtype=tf.float32)
        
        neigh_h = neigh_vecs
        


        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        
        neigh_vecs = tf.cast(neigh_vecs, dtype=tf.float32)
        self_vecs = tf.cast(self_vecs, dtype=tf.float32)
        
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim_1,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                 output_dim=hidden_dim_2,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))


        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        
        neigh_vecs = tf.cast(neigh_vecs, dtype=tf.float32)
        self_vecs = tf.cast(self_vecs, dtype=tf.float32)

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class TemporalAttentionLayer(Layer):
    """ The input parameter num_time_steps is set as the total number of training snapshots +1."""
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.bias = bias
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        self.input_dim = input_dim
        xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name + '_vars'):
            if use_position_embedding:
                self.vars['position_embeddings'] = tf.get_variable('position_embeddings',
                                                                   dtype=tf.float32,
                                                                   shape=[self.num_time_steps, input_dim],
                                                                   initializer=xavier_init)  # [T, F]

            self.vars['Q_embedding_weights'] = tf.get_variable('Q_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]
            self.vars['K_embedding_weights'] = tf.get_variable('K_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]
            self.vars['V_embedding_weights'] = tf.get_variable('V_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]

    def __call__(self, inputs):
        """ In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]."""
        # 1: Add position embeddings to input
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.num_time_steps), 0), [tf.shape(inputs)[0], 1])
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embeddings'],
                                                          position_inputs)  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = tf.tensordot(temporal_inputs, self.vars['Q_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        k = tf.tensordot(temporal_inputs, self.vars['K_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        v = tf.tensordot(temporal_inputs, self.vars['V_embedding_weights'], axes=[[2], [0]])  # [N, T, F]

        # 3: Split, concat and scale.
        q_ = tf.concat(tf.split(q, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        k_ = tf.concat(tf.split(k, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        v_ = tf.concat(tf.split(v, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        
        outputs = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)

        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = tf.ones_like(outputs[0, :, :])  # [T, T]
        tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [hN, T, T]
        padding = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), padding, outputs)  # [h*N, T, T]
        outputs = tf.nn.softmax(outputs)  # Masked attention.
        self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = tf.layers.dropout(outputs, rate=self.attn_drop)
        outputs = tf.matmul(outputs, v_)  # [hN, T, C/h]

        split_outputs = tf.split(outputs, self.n_heads, axis=0)
        outputs = tf.concat(split_outputs, axis=-1)

        # Optional: Feedforward and residual
        if FLAGS.position_ffn:
            outputs = self.feedforward(outputs)

        if self.residual:
            outputs += temporal_inputs

        return outputs

    def feedforward(self, inputs, reuse=None):
        """Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        """
        with tf.variable_scope(self.name + '_vars', reuse=reuse):
            inputs = tf.reshape(inputs, [-1, self.num_time_steps, self.input_dim])
            params = {"inputs": inputs, "filters": self.input_dim, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs += inputs
        return outputs