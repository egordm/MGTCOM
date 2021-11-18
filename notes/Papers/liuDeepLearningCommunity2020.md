---
type: paper
title: "Deep Learning for Community Detection: Progress, Challenges and Opportunities"
author: "Liu et al."
creator: "Egor Dmitriev (6100120)"

---

# Deep Learning for Community Detection: Progress, Challenges and Opportunities - Liu et al. 



## Goals

- A survey of current progress in community detection through deep learning
- Structured into three broad research streams in this domain
  - Deep neural networks
  - Deep graph embedding
  - Graph neural networks

## Preliminaries

- Communities: From the perspective of **connectedness and density**, communities are known as locally dense connected subgraphs or clusters of nodes 
- Deep Learning Advantages:
  - Ability to encode feature representations of **high-dimensional data**
  - Deep learning models can also **learn the pattern** of nodes, neighborhoods, and subgraphs
  - Deep learning is the superior choice for **unsupervised** learning tasks
  - **performance** improvements
  - the capacity to base detection on more and **richer features**;

## Challenges

- An Unknown Number of Communities:
  - [Bhatia and Rani, 2018] based on random walk-based personalized PageRank. However, this type of method cannot guarantee that every node in the network is assigned to a community
- Network Heterogeneity:
  - Network heterogeneity refers to networks that contain significantly different types of entities and relationships, which means the strategies used for homogeneous networks do not necessarily work. 
- Large-scale Networks:
  - Today, large-scale networks can contain millions of nodes, edges, and structural patterns and can also be highly dynamic, as networks like Facebook and Twitter demonstrate.

## Previous Work / Citations

- In **conventional machine learning**, detecting communities has generally been conceived as a **clustering problem on graphs**. But these approaches are highly dependent on the characteristics of the data
- Auto-Encoders:
  - Discovery that auto-encoders and spec-tral clustering have similar frameworks in terms of a low-dimensional approximation of the spectral matrix 
  - [Bhatia and Rani, 2018]
    - A random walk-based personalized PageRank and fine-tunes the detection by optimizing the modularity of the community structure
    - To avoid the need to preset the number of communities, a layer-wise stacked auto-encoder can effectively find centers of communities based on the network structure
  - [Cao et al., 2018a] 
    - Developed a graph regularized auto-encode
- GAN-based Approaches
  - fast-adjusting training precision
  - [Jia et al., 2019] 
    - Argued that traditional graph clustering-based community detection methods cannot handle the dense overlapping of communities
    - CommunityGAN that jointly solves overlapping community detection and graph representation learning based on GANs
- Spatial GCN
  - Pairwise relations to avoid searching a sparse adjacency matrix
  - [Xie et al., 2018]
    - An unsupervised deep learning algorithm extracts the network features, which are then used to partition the network

## Code

- ...

## Resources

- ...