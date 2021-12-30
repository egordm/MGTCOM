import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('base_model', 'SageDy', 'Base model string.')
flags.DEFINE_string('model', 'default', 'Model string.')

flags.DEFINE_string('dataset', 'ml-10m_new', 'Dataset string.')
flags.DEFINE_integer('time_steps', 10, '# time steps to train (+1)') # Predict at next time step.
flags.DEFINE_integer('GPU_ID', 0, 'GPU_ID')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 512, 'Batch size (# nodes)')
flags.DEFINE_boolean('featureless', True, 'Use 1-hot instead of features')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')

flags.DEFINE_integer('test_freq', 1, 'Testing frequency')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency')

flags.DEFINE_integer('neg_sample_size', 10, 'number of negative samples')
flags.DEFINE_integer('walk_len', 40, 'Walk len')
flags.DEFINE_float('neg_weight', 1, 'Wt. for negative samples')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate for self-attention model.')

flags.DEFINE_float('spatial_drop', 0.1, 'attn Dropout (1 - keep probability).')
flags.DEFINE_float('temporal_drop', 0.5, 'ffd Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight for L2 loss on embedding matrix.')

flags.DEFINE_boolean('use_residual', False, 'Residual connections')

flags.DEFINE_integer('identity_dim', 128,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_string('aggregator_layer', 'sagedy', 'model names. See README for possible values.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('hidden1', 128, 'units in GCN first layer')
flags.DEFINE_integer('hidden2', 128, 'units in GCN second layer')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_string('structural_head_config', '16', 'Encoder layer config: # attention heads in each GAT layer')
flags.DEFINE_string('structural_layer_config', '128', 'Encoder layer config: # units in each GAT layer')

flags.DEFINE_string('temporal_head_config', '16', 'Encoder layer config: # attention heads in each GAT layer')
flags.DEFINE_string('temporal_layer_config', '128', 'Encoder layer config: # units in each GAT layer')

flags.DEFINE_boolean('position_ffn', True, 'Use position wise feedforward')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
flags.DEFINE_integer('seed', 7, 'Random seed')

flags.DEFINE_string('save_dir', "output", 'Save dir defaults to output/ within the base directory')
flags.DEFINE_string('log_dir', "log", 'Log dir defaults to log/ within the base directory')
flags.DEFINE_string('csv_dir', "csv", 'CSV dir defaults to csv/ within the base directory')
flags.DEFINE_string('model_dir', "model", 'Model dir defaults to model/ within the base directory')
flags.DEFINE_integer('window', -1, 'Window for temporal attention (default : -1 => full)')
