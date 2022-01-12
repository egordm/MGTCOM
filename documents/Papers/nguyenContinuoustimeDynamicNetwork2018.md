---
type: paper
title: Continuous-Time Dynamic Network Embeddings
author: Nguyen et al.
creator: Egor Dmitriev (6100120)
---

# Continuous-Time Dynamic Network Embeddings - Nguyen et al.

## Goals

- Describe a general framework for incorporating temporal information into network embedding methods
- Methods for learning time-respecting embeddings from continuous-time dynamic networks
- TLDR: Describes a temporal walk strategy

## Preliminaries

- ...

## Challenges

- **General & Unifying Framework**: general framework for incorporating temporal dependencies in node
  embedding and deep graph models that leverage random walks
- **Continuous-Time Dynamic Networks**: timedependent network representation for continuous-time dynamic networks
- **Effectiveness**: Must outperform baselines

## Previous Work / Citations

- Static Snapshot Graphs:
  - each static snapshot graph represents all edges that occur between a user-specified
    discrete-time interval (e.g., day or week)
  - Refs: [57, 59, 63, 64]
  - Drawbacks:
    - Noisy approximation on continuous time
    - Selecting appropriate granularity
- **This Work:**
  - Random Walks
  - Supports **graph streams** (edges come and go live)
  - Any work using random walks can benefit from the proposed methods

## Definitions

* **Continuous-Time Dynamic Network**: $G=\left(V, E_{T}, \mathcal{T}\right)$
  * $E_T$ edges at continuous times (actually events)
* **Temporal Walk**:  temporal walk represents a temporally valid sequence of edges traversed in increasing order of edge times
* **Temporal Neighborhood**: $\Gamma_{t}(v)=\left\{\left(w, t^{\prime}\right) \mid e=\left(v, w, t^{\prime}\right) \in E_{T} \wedge \mathcal{T}(e)>t\right\}$
  * Neighbors of a node $v$ at time $t$
  * Nodes may appear multiple times (multiple edge events)
* * 

## Outline / Structure

- Random Walks
  - Changes walk space $\mathbb{S}$ to $\mathbb{S}_T$
- **Goal**: $f: V \rightarrow \mathbb{R}^D$: mapping nodes in $G$ to $D$-dimensional **time-dependent feature representation**
  - For ml tasks such as link prediction
- **Temporal Walks**:
  - Require starting time (Randomly samples or from a randomly samples edge)
  - Edges from further time may be less predictive (so bias wisely)
  - Has min length $\omega$
- **Biasing the walks**
  - Unbiased: $Pr(e) = 1/N$
  - Biased: Used a distribution based on time
  - Favor newer edges: $\operatorname{Pr}(e)=\frac{\exp \left[\mathcal{T}(e)-t_{\min }\right]}{\sum_{e^{\prime} \in E_{T}} \exp \left[\mathcal{T}\left(e^{\prime}\right)-t_{\min }\right]}$
    - Exp dist with $t_min$ as starting time
- **Biasing Neighbor selection**: Uniform or Biased (bias for time difference for example)
  - Walk bias van be reused based on $\tau(v)$
- Temporal Context Windows:
  - Window count: $\beta=\sum_{i=1}^{k}\left|\mathcal{S}_{t_{i}}\right|-\omega+1$
    - Number of walks that can be derived from the window with size $\omega$
- node2vec approach for

<img src="nguyenContinuoustimeDynamicNetwork2018.assets/Screenshot_20211027_130947.png" alt="Screenshot_20211027_130947" style="zoom:67%;" /><img src="nguyenContinuoustimeDynamicNetwork2018.assets/Screenshot_20211027_131008.png" alt="Screenshot_20211027_131008" style="zoom:67%;" />

## Evaluation

- Try different versions of network (2 main components to swap)
- Baselines: node2vec [26], DeepWalk [52], and LINE [65]. ,
- Datasets from NetworkRepository [58].
- 

## Code

- https://github.com/LogicJake/CTDNE
- https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/ctdne-link-prediction.html?highlight=TemporalRandomWalk

## Resources

- ...
