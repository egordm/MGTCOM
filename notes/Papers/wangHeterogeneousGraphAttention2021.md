---
title: "Heterogeneous Graph Attention Network"
author: "Wang et al."
creator: Egor Dmitriev (6100120)

---

# Heterogeneous Graph Attention Network - Wang et al. 

## Goals

- We first propose a novel heterogeneous graph neural network based on the hierarchical attention, including node-level and semantic-level attentions

## Preliminaries

- ...

## Challenges

- Real-world graph usually comes with multi-types of nodes and edges
- Different node types may have different attributes

## Previous Work / Citations

- metapath2vec: randomwalk -> skipgram (context based) (negative sampling)
- hin2vec: max likelihood based on path count/probability approximation (but employs a negative sampling like approach)
  - Uses multiple prediction training tasks which learn the latent vectors of nodes and meta-paths simultaneously
- PME projects different types of node into the same relation space and conducts heterogeneous link prediction.
- **This Work:** 
  - We introduce node-level attention can learn the importance of meta-path based neighbors for each node in a heterogeneous graph and aggregate the representation of these meaningful neighbors to form a node embedding.
  - To address the challenge of meta-path selection and semantic fusion in a heterogeneous graph, we propose a novel semantic-level attention to automatically learn the importance of different meta-paths and fuse them for the specific task.

## Definitions

* **Semantic-level attention**: aims to learn the importance of each meta-path and assign proper weights to them.
* **Node-level attention**:  aims to learn the importance of meta-path based neighbors and assign different attention values to them
* **Heterogeneous network**: $H = \{V, E, A, R, \phi, \psi\}$
  - $v_i \in V$: vertices, $e_{ij} \in E$: edges
  - $\phi(v_i)$ : Node type, $\psi(e_{ij})$: Link type
  - $A_i^o$: Node attribute, $U^o_{ij}$: Link attribute
* **Meta-Path**: Path $o_1 \rightarrow^{l_1} o_2 \rightarrow^{l_2} ... o_m \rightarrow^{l_{m+1}} o_{m+1}$
  * Where $o$ and $l$ are node/link types
  * Carries semantics (composed relation)
  * Allows computing **multi-modal proximity**
* **Network embedding**: $\mathbf{\Phi} : V \rightarrow \mathbb{R}^{|V| \times d}$
* **Heterogenous network embedding**: $\{\mathbf{\Phi}_k : V \rightarrow \mathbb{R}^{|V_k| \times d}\}^K_{k=1}$
  * where $K$ is number of node types

$$
\begin{array}{cc}
\hline \text { Notation } & \text { Explanation } \\
\hline \Phi & \text { Meta-path } \\
\mathbf{h} & \text { Initial node feature } \\
\mathbf{M}_{\phi} & \text { Type-specific transformation matrix } \\
\mathbf{h}^{\prime} & \text { Projected node feature } \\
e_{i j}^{\Phi} & \text { Importance of meta-path based node pair }(i, j) \\
\mathbf{a}_{\Phi} & \text { Node-level attention vector for meta-path } \Phi \\
\alpha_{i j}^{\Phi} & \text { Weight of meta-path based node pair }(i, j) \\
\mathcal{N}^{\Phi} & \text { Meta-path based neighbors } \\
\mathrm{Z}_{\Phi} & \text { Semantic-specific node embedding } \\
\mathrm{q} & \text { Semantic-level attention vector } \\
w_{\Phi} & \text { Importance of meta-path } \Phi \\
\beta_{\Phi} & \text { Weight of meta-path } \Phi \\
\mathrm{Z} & \text { The final embedding } \\
\hline
\end{array}
$$



## Outline / Structure

- Node-level attention: node-level attention can learn the importance of meta-path based neighbors for each node in a heterogeneous graph and aggregate the representation of these meaningful neighbors to form a node
  embedding.
- Project node features into a common space: $h'_i = \mathbf{M}_{\phi_i} \cdot h_i$

### Node-level Embeddings

- Importance of meta-path based node pair (**node-level attention**): $e^{\Phi}_{ij} = att_{node}(h_i', h_j'; \Phi)$
  - $att_{node}$ is MLP and is shared
  - $e^{\Phi}_{ij}$ is asymmetric; node level attention can preserve asymmetry
- Inject structural information via **masked attention** (by calculating neighbor weights):
  - $\alpha_{i j}^{\Phi}=\operatorname{softmax}_{j}\left(e_{i j}^{\Phi}\right)=\frac{\exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{j}^{\prime}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}^{\Phi}} \exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{k}^{\prime}\right]\right)\right)}$
    - $\sigma$: activation
    - $||$ : concatenation operation
    - Referred to as Weight Coefficient of Meta-path node pair
- Meta-path based embedding: aggregated based on neighbors and their weight coefs:
  - $\mathbf{z}_{i}^{\Phi}=\sigma\left(\sum_{j \in \mathcal{N}_{i}^{\Phi}} \alpha_{i j}^{\Phi} \cdot \mathbf{h}_{j}^{\prime}\right)$

Close up of meta-path based embedding calculation

<img src="wangHeterogeneousGraphAttention2021.assets/Screenshot_20211102_211729.png" alt="Figure 3 Explanation of aggregating process in both node-level and semantic-level." style="zoom:80%;" />

* Extend node-level attention to multi-head attention:
  * Since heterogeneous graph present the property of scale free, the variance of graph data is quite high
  * Process becomes more stable
  * Repeat node-level attention for $K$ times
  * $\mathbf{z}_{i}^{\Phi}=||_{k=1}^{K} \sigma\left(\sum_{j \in N_{i}^{\Phi}} \alpha_{i j}^{\Phi} \cdot \mathbf{h}_{j}^{\prime}\right)$

### Semantic-level attention

* Calculate weight of each meta-path node pair: $\left(\beta_{\Phi_{1}}, \ldots, \beta_{\Phi_{P}}\right)=a t t_{\text {sem }}\left(\mathrm{Z}_{\Phi_{1}}, \ldots, \mathrm{Z}_{\Phi_{P}}\right)$
* $att_{sem}$: DNN performing semantic level attention
  * Calculate importance of a meta path (by averaging the pair weights)
    * $w_{\Phi_{p}}=\frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{q}^{\mathrm{T}} \cdot \tanh \left(\mathbf{W} \cdot \mathbf{z}_{i}^{\Phi_{p}}+\mathbf{b}\right)$
  * Weight of meta-path is obtained by normalizing the importance of all meta-paths
    * $\beta_{\Phi}=\frac{\exp \left(w_{\Phi_{p}}\right)}{\sum_{p=1}^{P} \exp \left(w_{p}\right)}$
    * can be interpreted as the contribution of the meta-path $\Phi_p$ for specific task
* Final embedding: 
  * $\mathrm{Z}=\sum_{p=1}^{P} \beta_{\Phi_{p}} \cdot \mathrm{Z}_{\Phi_{p}}$
* Loss function: minimizing Cross-Entropy over all labeled nodes

<img src="wangHeterogeneousGraphAttention2021.assets/Screenshot_20211102_210011.png" alt="Screenshot_20211102_210011" style="zoom:80%;" />



<img src="wangHeterogeneousGraphAttention2021.assets/Screenshot_20211102_220150.png" alt="Overall algorithm" style="zoom:80%;" />

## Evaluation

- Does outperform other algs by a (substantial) margin.
- Uses: DBLP, ACM, IMDB 
- Evaluates node classification task
- Evaluates for clustering task (apply K-Means afterwards)

## Code

- https://github.com/Jhy1993/HAN
  - Code is a little messy
  - $a^{\Phi}_{ij}$ : https://github.com/Jhy1993/HAN/blob/71bac29a07fb8fab908d50a806a7bc38aa6c6611/models/gat.py#L43
  - $w_{\Phi}$: https://github.com/Jhy1993/HAN/blob/71bac29a07fb8fab908d50a806a7bc38aa6c6611/models/gat.py#L68
  - semantic level attention calculation ?
    - https://github.com/Jhy1993/HAN/blob/71bac29a07fb8fab908d50a806a7bc38aa6c6611/models/gat.py#L61

## Resources

- ...

## Discussion

* That algorithms seems so compute heavy
  * Especially that softmax (which is not adressed in analysis?)
* Query based attention / importance would be cool
  * But that is a transformer like thing
* 

