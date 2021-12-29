---
type: paper
title: A Comprehensive Survey on Graph Neural Networks
author: Wu et al.
creator: Egor Dmitriev (6100120)
---

# A Comprehensive Survey on Graph Neural Networks

## Goals

- **New taxonomy**: We propose a new taxonomy to divide the state-of-the-art graph neural networks into four categories, namely **recurrent graph neural networks**, **convolutional graph neural networks**, **graph autoencoders**, and **spatial-temporal graph neural networks**
- **Comprehensive review**: We provide the most comprehensive overview of modern deep learning techniques for graph data
- **Abundant resources**: We collect abundant resources on graph neural networks, including state-of-the-art models, benchmark data sets, open-source codes, and practical applications
- **Future directions**: We discuss theoretical aspects of graph neural networks, analyze the limitations of existing methods, and suggest four possible future research directions in terms of model depth, scalability trade-off, heterogeneity, and dynamicity

## Definitions

- ...

## Challenges

- As graphs can be irregular, a graph may have a variable size of unordered nodes, and nodes from a graph may have a different number of neighbors, resulting in some important operations
- a core assumption of existing machine learning algorithms is that instances are independent of each other. This assumption no longer holds for graph data because each instance (node) is related to others by links of various types, such as citations, friendships, and interactions

## Outline / Structure

- **Recurrent graph neural networks (RecGNNs)**
  - RecGNNs aim to learn node representations with recurrent neural architectures. They assume a node in a graph constantly exchanges information/message with its neighbors until a stable equilibrium is reached.
- **Convolutional graph neural networks (ConvGNNs)** 
  - generalize the operation of convolution from grid data to graph data. The main idea is to generate a node v’s representation by aggregating its own features xv and neighbors’ features $x_u$, where $u \in N (v)$.
- **Graph autoencoders (GAEs)**
  - are unsupervised learning frameworks which encode nodes/graphs into a latent vector space and reconstruct graph data from the encoded information
- **Spatial-temporal graph neural networks (STGNNs)** 
  - aim to learn hidden patterns from spatial-temporal graphs
- **Graph Attention Network (GAT)**
  - assumes contributions of neighboring nodes to the central node are neither identical like GraphSage [42], nor pre-determined like GCN
  - GAT further performs the multi-head attention to increase the model’s expressive capability. This shows an impressive improvement over GraphSage on node classification tasks.
* Spectral-based ConvGNNs
  * Note paper has some really intuitive explaination on spectral based GNNs

### Theoretical Foundation

* **Shape of receptive field**: The receptive field of a node is the set of nodes that contribute to the determination of its final node representation.
* **VC dimension**:  The VC dimension is a measure of model complexity defined as the largest number of points that can be shattered by a model. This result suggests that the model complexity of a GNN* [15] increases rapidly with p and n if the sigmoid or tangent hyperbolic activation is used
* **Graph isomorphism**:  Two graphs are isomorphic if they are topologically identical. They show that common GNNs such as GCN [22] and GraphSage [42] are incapable of distinguishing different graph structures. Xu et al. [57] further prove if the aggregation functions and the readout functions of a GNN are injective, the GNN is at most as powerful as the WL test in distinguishing different graphs.
* **Equivariance and invariance**:  In order to achieve equivariance or invariance, components of a GNN must be invariant to node orderings
* **Universal approximation**:   Xu et al. [57] show that ConvGNNs under the framework of message passing [27] are not universal approximators of continuous functions defined on multisets. Maron et al. [104] prove that an invariant graph network can approximate an arbitrary invariant function defined on graphs.
  
  

### **Spatial-temporal graph neural networks (STGNNs)**

- Methods under this category aim to model the dynamic node inputs while assuming interdependency between connected nodes. 
- Most **RNN-based approaches** capture spatial-temporal dependencies by filtering inputs and hidden states passed to a recurrent unit using graph convolutions [48], [71], [72].
- Diffusion Convolutional Recurrent Neural Network (DCRNN) [72] incorporates a proposed diffusion graph convolutional layer (Equation 18) into a GRU network. In addition, DCRNN adopts an encoder-decoder framework to predict the future K steps of node values.
- **RNN-based approaches** suffer from time-consuming iterative propagation and gradient explosion/vanishing issues.
- **CNN-based approaches** tackle spatial-temporal graphs in a non-recursive manner with the advantages of parallel computing, stable gradients, and low memory requirements.
- Learning latent static spatial dependencies can help researchers discover interpretable and stable correlations among
  different entities in a network. 

### Notes

* The paper does not include much about **dynamic** networks
