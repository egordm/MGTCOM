#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 22:38
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : config.py
# @Software: PyCharm
# @Describe:
import argparse
import os

def random_string():
    """Generate a random string."""
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(10))


parser = argparse.ArgumentParser()
parser.add_argument('-n', default=0, type=int, help='GPU ID')
parser.add_argument('--dataset', type=str, default='StarWars')
parser.add_argument('--dataset_version', type=str, default='base')
parser.add_argument('--run_name', type=str, default=random_string())
parser.add_argument('--primary_type', type=str, default='Character')
parser.add_argument('--repr_dim', type=int, default=64)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=1024 * 80)
parser.add_argument('--k', type=int, default=20)
args = parser.parse_args()

config_path = os.path.dirname(__file__)
data_config = {
    'data_path': os.path.join(config_path, 'data'),
    'dataset': args.dataset,
    'dataset_version': args.dataset_version,
    'run_name': args.run_name,
    'primary_type': args.primary_type,
    'task': ['CF', 'CL'],
    'K_length': 4,
    'resample': False,
    'random_seed': 123,
    'test_ratio': 0.2
}

model_config = {
    'primary_type': data_config['primary_type'],
    # 'auxiliary_embedding': 'non_linear',  # auxiliary embedding generating method: non_linear, linear, embedding
    'auxiliary_embedding': 'emb',  # auxiliary embedding generating method: non_linear, linear, embedding
    'K_length': data_config['K_length'],
    'embedding_dim': args.repr_dim,
    'in_dim': args.hid_dim,
    'out_dim': args.hid_dim,
    'num_heads': 8,
    'merge': 'linear',  # Multi head Attention merge method: linear, mean, stack
    'g_agg_type': 'mean',  # Graph representation encoder: mean, sum
    'drop_out': 0.3,
    'cgnn_non_linear': True,  # Enable non linear activation function for CGNN
    'multi_attn_linear': False,  # Enable atten K/Q-linear for each type
    'graph_attention': True,
    'kq_linear_out_dim': 128,
    'path_attention': False,  # Enable Context path attention
    'c_linear_out_dim': 8,
    'enable_bilinear': False,  # Enable Bilinear for context attention
    'gru': True,
    'add_init': False
}

train_config = {
    'continue': False,
    'lr': 0.05,
    'l2': 0,
    'factor': 0.2,
    'total_epoch': 10000000,
    # 'batch_size': 1024 * 20,
    'batch_size': args.batch_size,
    'pos_num_for_each_hop': [5, 5, 5, 5, 5, 5, 5, 5, 5],
    'neg_num_for_each_hop': [3, 3, 3, 3, 3, 3, 3, 3, 3],
    'sample_workers': 8,
    'patience': 5,
    'checkpoint_path': os.path.join(config_path, 'checkpoint', data_config['dataset'])
}

evaluate_config = {
    'method': 'LR',
    'save_heat_map': True,
    'result_path': os.path.join('result', data_config['dataset']),
    'random_state': 123,
    'max_iter': 500,
    'n_jobs': 1,
}
