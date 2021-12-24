---
type: paper
title: "Community-aware dynamic network embedding by using deep autoencoder"
subtitle: "By Ma et al"
author:
  name: "Egor Dmitriev (6100120)"
  institute: "Utrecht University"
  country: "The Netherlands"
  email: "e.dmitriev@students.uu.nl"

---

# Goals

- Propose a Community-aware Dynamic Network Embedding method (CDNE)
  - aiming to learn the low-dimensional representations of nodes over time while preserving the global node structures, local link structures, and continuous community dynamics
  - validate the superiority of CDNE over several state-of-the-art DNE methods

# Preliminaries

- **Network Embedding**:  technique for learning the low-dimensional representation of entities while preserving their properties
- Generally, the **high-dimensional** structures are represented by:
  - the **microscopic structures** such as the link proximity [41], cycles, subspaces [44], paths [32] and multilayered links [22] 
  - the **macroscopic structures** such as the community structures and subgraphs
- Many real-world networks such as collaboration networks, social networks and biological networks are naturally dynamic, with the emergence and disappearance of nodes and edges over time
- 

# Challenges

- Smooth dynamic assumption in [7,11,12,14] does not always hold as the **microscopic node and link structures of some systems may evolve sharply**
- 

# Previous Work / Citations

- Some dynamic NE (DNE) methods have been proposed under the assumption that the **microscopic node and link structures** constantly change over time: (AKA **dynamic node representation learning**)
  - Goyal et al. [12,14] proposed to learn the low-dimensional representations of nodes incrementally:
    - Representations at current step depending those on previous time steps
    - Proposed Dyn2vec and DySAT, which use a self-attention technique and a vector recurrent technique to learn the potential structures of dynamic networks at **multiple non-linear time stamps**, respectively [11]. 
  - Du et al. [7] learned the smooth evolution of the most influential nodes
  - DHPE [50] imposed some extra feature information into dynamic networks
  - Drawbacks:
    - This assumption neglects the dynamics of the macroscopic community structures in dynamic network
    - Microscopic node and link structures of some systems may evolve sharply
- **This Work:** 
  - We present a novel dynamic network embedding method (called as CDNE)
  - With **consideration of the dynamics** of the **microscopic** node structures **and macroscopic** community structures
  - To preserve the dynamics of communities, we present a **community-aware smoothing technique** and **adopt a *temporal regularizer***.
  - We use a combination of the ***first-order proximity*** and the ***second-order proximity*** of nodes in dynamic networks
  - We adopt a ***stacked deep autoencoder*** algorithm to learn the low-dimensional representations of nodes
  - Intuition:
    - Communities generally have a smoother evolution than nodes and links in dynamic systems
    - Embedding representations of a community at adjacent time should be close to each other in the low-dimensional space

# Definitions

* **Dynamic network**: $\mathcal{G}$ is a series of network snapshots
* **Adjacency matrix**: $A^a$ Represents a snapshot at time $a$
* **Community Structure**:  A community in a dynamic network consists of a set of nodes, such that the nodes in the community are densely linked with each other whereas they are sparsely connected with the other nodes *at all time stamps*
  * communities have a smoother evolution than nodes and links in dynamic systems
* **Dynamic Network Embedding (DNE)**: 
  * Finds  a time-series of mappings $\mathbf{F = \{F^1, F^2 , . . . F^t \}}$, and each mapping $F^a: S^a \rightarrow H^a$ , learns the latent $d$ dimensional representations $H^a$ of nodes at $a$th time stamp such that the learned representations can effectively preserve both the **structural information** $S = \{S^1 , S^2 , . . . , S^t \}$ and the evolutionary pattern in $\mathcal{G}$.
  
  ![](/dd_volume/Development/Python/Thesis/documents/Papers/maCommunityawareDynamicNetwork2020.assets/Screenshot_20211214_215516.png)
  
  

# Outline / Structure

## **Global structure preservation**

- Global structure of a network reflects the **similarities of nodes** which are usually evaluated by their link structures.
- Evaluating by link structures neglects the similarities of two unlinked nodes
- **Use a combination** of the **first-order proximity** $S^1$ and the **second-order proximity** $S^2$ to represent the global structure of networks
  - **first-order proximity** $S^1$: evaluates the number of direct neighbor of nodes
  - **second-order proximity** $S^2$:  measures the similarity of the common neighbors of node
  - Global structure $S^a$: $S^{a}=S^{1, a}+\lambda S^{2, a}$
- **Global Reconstruction Error**: $\mathbf{L}_{g}^{a}=\sum_{i=1}^{n}\left\|\hat{S}_{i .}^{a}-S_{i .}^{a}\right\|_{2}^{2}$ where
  - Preserves the global structures by adopting objective of a stacked deep autoencoder 
  - $S$:  **true global structures**
  - $\hat{S} = D(F(S))$: structures **decoded by the decoder** $D$ from the low-dimension representations generated by the encoder $F$ for the structures $S$.

## Local Structure Preservation

* Local structure of a node is the neighbors of the node
* From the view of social ***homophily***, the neighbors of an individual $i$ are a set of individuals that are highly connected with $i$.
* **Local Loss**: $\mathbf{L}_{l}^{a}=\sum_{e_{i j} \in \mathcal{E}^{a}}\left\|\mathrm{H}_{i}^{a}-\mathrm{H}_{j .}^{a}\right\|_{2}^{2}$  where
  * $H^a_i$. is the low-dimension representation of node $i$ at $a$th timestamp

## Evolution community preservation

* **Community** in a dynamic network G is composed of a set of nodes which are **densely linked** with each other and **have similar structure functions** at most time stamps
  * **Microscopic** node structures temporally change over time
  * **Macroscopic** community structures smoothly evolve over time
  * Node may sharply change links, but its community evolves smoothly
* **Community Evolution Loss**: $\mathbf{L}_{c}^{a}= \begin{cases}\sum_{k=1}^{q} \sum_{i \in \mathcal{C}_{k}}\left\|\mathrm{H}_{i}^{a}-\mathrm{H}_{c_{k}}^{a-1}\right\|_{2}^{2} & \text { if } a>1 \\ 0 & \text { if } a=0\end{cases}$
  * Minimizes the Euclidean distance between nodes and their communities in the low-dimensional representations at adjacent time stamps
  * Allows for **maintaining community stability**
  * $\mathrm{H}_{c_{k}}^{a}$: is the low-dimensional representation of community $C_k$ at $a$th time stamp
    * $\mathrm{H}_{\mathcal{C}_{k}}^{a}=\frac{\sum_{i \in \mathcal{C}_{k}} \mathrm{H}_{i .}^{a}}{\left|\mathcal{C}_{k}\right|}$ is average of its members’ representations
* **Tradeoff between the sensibility and stability** (Small Scale vs Large Scale communities)
  * In the embedded space, a **node is closer to its small-scale community than its large-scale community**
  * Small-scale community is more sensitive to the evolution than a large-scale community
* **Community Evolution Loss (Hierarchical)**: $\mathbf{L}_{c}^{a}= \begin{cases}\sum_{k=1}^{q} \sum_{j=1}^{q_{k}} \sum_{i \in \mathcal{C}_{k}^{j}}\left\|\mathrm{H}_{i .}^{a}-\mathrm{H}_{\mathcal{C}_{k}^{j}}^{a-1}\right\|_{2}^{2} & \text { if } a>1 \\ 0 & \text { if } a=0\end{cases}$ where
  * Community: $\mathcal{C}_{k}=\left\{\mathcal{C}_{k}^{1}, \mathcal{C}_{k}^{2}, \ldots, \mathcal{C}_{k}^{q_{k}}\right\}$  
  * $q_k$ is the number of small scale communitiies
  * $\mathrm{H}_{\mathcal{C}_{k}^{j}}^{a}=\frac{\sum_{i \in \mathcal{C}_{k}^{j}} \mathrm{H}_{i}^{a}}{\left|\mathcal{C}_{k}^{j}\right|}$ 
* Number of subcommunities: $q_k$ 
  * Determined by $w$ (hyperparameter) which controls the size of small communities
* Calculating the loss requires already having communities:
  * **Requires having some prior knowledge about community structures**
  * Communities are **not known a priori**
  * Therefore they are **initialized using existing CD algorithms** (classical)
    * Large scale communities: Genlouvin (requires no priors)
      * More rough but very fast
    * Small scale communities: k-means (prior is $w$ community size)
      * More fine tuned

## Objective

* Given a dynamic network $\mathcal{G}$ and dimension size $d$
  * Find optimal encoders: $\mathbf{F = \{F^1, F^2 , . . . F^t \}}$
    * Which maps structure $S^a$ to a (low) $d$-dimensional representation $H^a$
  * Find optimal decoders: $\mathbf{D = \{D^1, D^2 , . . . D^t \}}$
    * Which maps structure $H^a$ to a high-dimensional network representation $\hat{S}^a$
* Unified loss: $\mathbf{L}^{a}=\mathbf{L}_{g}^{a}+\alpha \cdot \mathbf{L}_{l}^{a}+\beta \cdot \mathbf{L}_{c}^{a}$
  * $\alpha$ affects the performance of our CDNE on the **prediction** tasks
  * $\beta$ has significant impacts on the **stabilization** tasks
  * $\mathbf{L}_{l}^{a}$ and $\mathbf{L}_{c}^{a}$ are referred to as regularizers
  * 

## Architecture

* Employs quite simplistic Spectral GAE architecture
* Encoder: $\mathrm{H}_{i .}^{a, o}=\mathbf{F}^{a, o}\left(\mathrm{H}_{i .}^{a, o-1}\right)=\sigma\left(\mathrm{W}^{a, o} \cdot \mathrm{H}_{i .}^{a, o-1}+\mathrm{b}^{a, o}\right)$
  * Note that $\mathrm{H}_{i .}^{a, 0} = S^a$
  * $o$ is the layer number
  * $S^a = \hat{S}^{a} = \left[ \hat{S}_{i j}^{a} \right]_{n \cdot n} \in \mathbb{R}^{n \times n}$  kinda Laplacian matrix but different
  * Uses **Unified Loss** $L^a$
* Decoder: $\hat{S}_{i .}^{a, o}=\mathbf{D}^{a, o}\left(\mathrm{H}_{i .}^{a, o}\right)=\sigma\left(\hat{\mathrm{W}}^{a, o} \cdot \mathrm{H}_{i .}^{a, o}+\hat{\mathrm{b}}^{a, o}\right)$
  * Maps back to the “kinda” Laplacian matrix
  * Uses only **Reconstruction Loss** $L^a_g$

![](/dd_volume/Development/Python/Thesis/documents/Papers/maCommunityawareDynamicNetwork2020.assets/Screenshot_20211214_223640.png)

# Evaluation

- Datasets:
  - Synthetic: 
    - SYN @ghalebiDynamicNetworkModel2019
    - SBM @lancichinettiBenchmarksTestingCommunity2009
  - Real:
    - ia-contacts, ia-email, ia-enron, ia-stackexch - @rossiNetworkDataRepository2015
    - TCGA, ca-cit-HepTh 
    - soc-wiki-elec  - Wikipedia Adminitrat
    - football
- Baselines: - Mainly Dynamic Network Embedding models
  - SDNE - structural deep autoencoder  for static networks
  - M-NMF - modularized nonnegative matrix factorization (NMF)
  - SAGE - GraphSAGE - not 
  - DynGEM - It uses a deep autoencoder to solve the DNE problem in dynamic networks
  - Dyn2vecAE - Capturing network dynamics using dynamic graph representation learning
- Criteria:
  - **Network reconstruction**:  It is used to test the performance of NE methods on reconstructing the link structures of networks
    - Average reconstruction precision (pr)
  - **Link prediction**:
    - For dynamic networks, it is adopted to predict the existence of links in the next time stamp
  - **Network stabilization**:
    - Performance of DNE methods on the stabilization of embedding
    - Dynamic network should have similar evolutionary patterns in both the learned low-dimensional representation and the network representation over time
  - **Community stabilization**: 
    - Stability of communities in dynamic networks on the embedded low-dimensional representations.
      
      

# Discussion

* A lot of small mistakes in the paper (as if it is a prepublication - it is not)
* There is no source code
* The baselines are not DCD algorithms but Dynamic Network Embedding algorithms
* Community memberships can’t change?
* This, and @wangEvolutionaryAutoencoderDynamic2020
  * Use datasets with very few nodes
  * Possibly because of the gigantic matrices for reconst

# Code

- ...

# Resources

- ...
