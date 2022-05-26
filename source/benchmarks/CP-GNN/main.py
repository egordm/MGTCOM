#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 10:06
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : main.py
# @Software: PyCharm
# @Describe:
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler

import torch
import dgl
from models import ContextGNN
from evaluate import evaluate_task
from utils import load_data, EarlyStopping, load_latest_model, evaluate, ExtGraphDataLoader, ROOT_PATH
import argparse

# parser = argparse.ArgumentParser(description='Which GPU to run?')
# parser.add_argument('-n', default=0, type=int, help='GPU ID')
# args = parser.parse_args()


def main(config):
    dataset = config.data_config['dataset']
    dataset_version = config.data_config['dataset_version']
    run_name = config.data_config['run_name']

    config.data_config['K_length'] -= 2
    # dataloader = ExtGraphDataLoader(Path('/data/pella/projects/University/Thesis/Thesis/source/storage/exports/StarWars/base'))
    dataloader = ExtGraphDataLoader(
        dataset, dataset_version, config.data_config
    )
    data = dataloader.data
    # dataloader = load_data(config.data_config)
    hg = dataloader.heter_graph
    edges_data_dict = dataloader.load_train_k_context_edges(hg, config.data_config['K_length'],
                                                            config.data_config['primary_type'],
                                                            config.train_config['pos_num_for_each_hop'],
                                                            config.train_config['neg_num_for_each_hop'])

    # CF_data = dataloader.load_classification_data()
    # LP_data = dataloader.load_links_prediction_data()
    LP_data = None
    # device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    dataloader_dict = {key: DataLoader(dataset, batch_size=config.train_config['batch_size'],
                                       # num_workers=config.train_config['sample_workers'], collate_fn=dataset.collate,
                                       num_workers=0, collate_fn=dataset.collate,
                                       shuffle=True, pin_memory=True)
                       for
                       key, dataset in
                       edges_data_dict.items()
                       if len(dataset) > 0}

    model = ContextGNN(hg, config.model_config)
    model = model.to(device)

    if config.train_config['continue']:
        model = load_latest_model(config.train_config['checkpoint_path'], model)

    stopper = EarlyStopping(checkpoint_path=config.train_config['checkpoint_path'], config=config,
                            patience=config.train_config['patience'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train_config['lr'], weight_decay=config.train_config['l2'])  # torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.train_config['factor'], patience=config.train_config['patience'] // 3, verbose=True)


    print("Start training...")
    for epoch in range(config.train_config['total_epoch']):
        running_loss = []
        for k_hop, dataloader in dataloader_dict.items():  # k_hop in [1, K+1]
            # print("Training %d k_length..." % k_length)
            for pos_src, pos_dst, neg_src, neg_dst in dataloader:
                p_context_emb = model(k_hop)
                p_emb = model.primary_emb.weight
                loss = model.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
                running_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        mean_loss = mean(running_loss)
        scheduler.step(mean_loss)  # Reduce learning rate
        print("Epoch:{}/{} Loss: {}".format(epoch, config.train_config['total_epoch'], mean_loss))
        early_stop = stopper.step(mean_loss, model)
        p_emb = model.primary_emb.weight.detach().cpu().numpy()
        evaluate(p_emb, None, LP_data, config.evaluate_config['method'], metric=config.data_config['task'],
                 random_state=config.evaluate_config['random_state'],
                 max_iter=config.evaluate_config['max_iter'], n_jobs=config.evaluate_config['n_jobs'])
        if early_stop:
            break
    checkpoint_path = stopper.filepath
    # evaluate_task(config, checkpoint_path)
    model = stopper.load_checkpoint(model)
    p_emb = model.primary_emb.weight
    a_emb = model.embedding_transformer(p_emb)
    # emb_dict = {
    #     k: v.detach().cpu().numpy()
    #     for k, v in a_emb.items()
    # }
    emb_dict = {
        k: a_emb[k].detach().cpu().numpy()
        for k in data.node_types
    }
    emb = torch.cat(list(a_emb.values()), dim=0).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=20, random_state=0).fit(emb)
    mus = kmeans.cluster_centers_
    # z_dict = {
    #     k: kmeans.predict(v_emv)
    #     for k, v_emv in emb_dict.items()
    # }
    z_dict = {
        k: kmeans.predict(emb_dict[k])
        for k in data.node_types
    }

    outputs_path = ROOT_PATH / "storage" / "outputs"
    output_dir = outputs_path / 'CP-GNN' / dataset / dataset_version / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'embeddings.npy', emb_dict)
    np.save(output_dir / 'assignments.npy', z_dict)
    np.save(output_dir / 'means.npy', mus)
    print(run_name)

    # result_save_path = evaluate(p_emb, CF_data, LP_data, method=config.evaluate_config['method'], save_result=True,
    #                             random_state=config.evaluate_config['random_state'],
    #                             max_iter=config.evaluate_config['max_iter'], n_jobs=config.evaluate_config['n_jobs'])
    # if result_save_path:
    #     save_config(config, result_save_path)
    #     attention_matrix_path = save_attention_matrix(model, result_save_path, config.data_config['k_length'])
    #     if attention_matrix_path:
    #         generate_attention_heat_map(hg.ntypes, attention_matrix_path)
    return


if __name__ == "__main__":
    import config

    main(config)
