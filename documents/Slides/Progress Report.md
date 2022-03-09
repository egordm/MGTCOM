# Progress Report (Sprint 4/10)

## Previous Meetings Recap

* Dataset selection and preprocessing
* Dataset analysis
* Implementation of evaluation metrics
* Ran benchmarks on
  * Link-based community detection
  * Representation-based community detection
* Proof of concept implementation
  * node2vec
  * graphSAGE



## Overview Architecture

### Stage 1: Positional Clustering

* $Z_n \in \mathbb{R}^{n \times d}$  (Positional Embeddings)
* $C_k \in \mathbb{R}^{k \times q}$ (Cluster Center Embeddings)
* $Q_n \in \mathbb{R}^{n \times k} = Z_n \cdot C_k^T$ (Cluster Assignment Representation)



<img src="file:///home/egordm/.config/marktext/images/2022-03-09-12-03-15-image.png" title="" alt="" width="792">



### Stage 2: Temporal Correction

* Freeze Positional Embeddings
* Use temporally aware neighbor sampling
  * Either window based
  * Or relative window based
* Train $Z_n^{temp}$ and $C_k^{temp}$
* Calculate loss over:
  * $Z_n = Z_n^{posi} || Z_n^{temp}$
  * $C_k = C_k^{posi} || C_k^{temp}$



<img src="file:///home/egordm/.config/marktext/images/2022-03-09-12-23-11-image.png" title="" alt="" width="580">



### What do we have?

* Stable Positional Representation
  * Can be inferred for new nodes given a neighborhood
* Variable Temporal Representation
  * Can be inferred for new nodes given a temporal neighborhood
* Static soft clustering
  * And their temporal variant
* Ability to infer cluster representation given a set of nodes
  * Both position based and temporal based
  * Also ability to infer cluster for a node
* Comparative static clustering performance to Louvain
  * Better performance than louvain for more fine grained clusters
    * Lower levels in hierarchy



## Progress Sprint 4

### Goals

* Implement temporal neighborhood sampling (instead of using multiple snapshots)
* Update proof of concept to support heterogeneous graphs
*  Experiment with various network architectures
  * To improve stability
  * Speedup training process



### Neighborhood Sampling

![](/home/egordm/.config/marktext/images/2022-03-09-10-12-47-image.png)



#### PinSAGE

R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and J. Leskovec, “Graph Convolutional Neural Networks for Web-Scale Recommender Systems,” in *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, New York, NY, USA, Jul. 2018, pp. 974–983. doi: [10.1145/3219819.3219890](https://doi.org/10.1145/3219819.3219890).

* For each *edge type* select $k_l$ neighbors for $l \in L$ layers 
* Uses **Personalized Page Rank (PPR)** to weigh neighboring nodes
* Actually uses **random walks** to approximate ppr
* Problems:
  * If there are many edge types the sampled neighbor count becomes large (but bounded)

<img src="file:///home/egordm/.config/marktext/images/2022-03-09-10-40-43-image.png" title="" alt="" width="509">



#### Heterogenous Graph Transformers (HGT)

Z. Hu, Y. Dong, K. Wang, and Y. Sun, “Heterogeneous Graph Transformer,” in *Proceedings of The Web Conference 2020*, Taipei Taiwan, Apr. 2020, pp. 2704–2710. doi: [10.1145/3366423.3380027](https://doi.org/10.1145/3366423.3380027).

<img src="file:///home/egordm/.config/marktext/images/2022-03-09-10-49-05-image.png" title="" alt="" width="953">



* Slightly modifies GraphSAGE sampling
  * Imposes a per node type budget per layer
  * Neighbors are weighted based on their normalized degree (weighted sampling)
* Fixes issue with heterogenous graphs (where the node count explodes)
* Problem:
  * Nodes cannot be sampled twice
    * Problematic when you want stateful temporal sampling
    * Only supports windowed temporal sampling



### tch_geometric

* **Rust** library with python bindings, containing implementation for neighborhood sampling
  
  
  
  #### Why?
* Neighborhood sampling can not be done batchwise (must be done per node individually)
  * Python is too slow for this
* Python is bad at multithreading - Global Interpreter Locking (GIL) :(
  * Memory limited possibilities for memory sharing between processes
  * Therefore a copy of the graph has to be maintained per process
* **torch_geometric** CSC and CSR implementations do not support parallel edges
  * COO (standard format) does but cant be used for sampling
* Standard implementations in **torch_geometric** do not implement **temporal sampling** or **weighted sampling**
* 



#### Graph Modelling

* CSR is used for outbound edges
  * Commonly used by Random Walk
* CSC is used for inbound edges
  * Used for neighborhood sampling for GraphSAGE

<img src="https://imgs.developpaper.com/imgs/1050201c7-2.gif" title="" alt="Instructions for using Python SciPy sparse matrix" width="583">

<img title="" src="https://imgs.developpaper.com/imgs/105020H29-3.gif" alt="Instructions for using Python SciPy sparse matrix" width="594">



#### Modelling Parallel edges

<img src="file:///home/egordm/.config/marktext/images/2022-03-09-11-14-07-image.png" title="" alt="" width="598">



#### Sampling Implementations

* Pinsage/GraphSAGE neighborhood sampling
  * Homogenous/Heterogenous graph support
  * Samping with/without replacement and weighted sampling support
  * Support for temporal sampling constraints
    * Windowed (all nodes are within a time range)
    * Relative (all neighbor nodes are within a time range from the **root node**)
    * Dynamic (all neighbors nodes are within a timerange from the **previous node**)
* HGT neighborhood sampling
  * Heterogenous graph support only
  * Sampling without replacement
  * Support for temporal sampling constraints
    * Windowed (described in original paper)
    * Relative (my addition) 
      * Implemented by recording edge states instead of node state



#### Other contributions

* tch-rs is a rust library with bindings to pytorch
  * Can be used to train pytorch models without python
  * Had no interop support for python

<img src="file:///home/egordm/.config/marktext/images/2022-03-09-11-36-41-image.png" title="" alt="" width="1018">





### Network Architecture and Importance Sampling

![](/home/egordm/.config/marktext/images/2022-03-09-10-29-59-image.png)

#### PinSAGE

R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and J. Leskovec, “Graph Convolutional Neural Networks for Web-Scale Recommender Systems,” in *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, New York, NY, USA, Jul. 2018, pp. 974–983. doi: [10.1145/3219819.3219890](https://doi.org/10.1145/3219819.3219890).

* PinSAGE
  * 
* Results:
  * Improves stability of the algorithm
  * Doesn't improve upper bound of the resulting clustering score
  
  * Need to do k random walks per node and assign weight to each edge
  * Heterogenous graphs are troublesome (lot of weights)

#### Heterogenous Graph Transformers (HGT)

Z. Hu, Y. Dong, K. Wang, and Y. Sun, “Heterogeneous Graph Transformer,” in *Proceedings of The Web Conference 2020*, Taipei Taiwan, Apr. 2020, pp. 2704–2710. doi: [10.1145/3366423.3380027](https://doi.org/10.1145/3366423.3380027).

![](/home/egordm/.config/marktext/images/2022-03-09-10-21-42-image.png)



* HGT
  * Uses Transformers (with multiple attention heads) to attend on important neighborhood nodes
  * Models relations $\langle\tau(s), \phi(e), \tau(t)\rangle$ by decomposing them into weight matrices
    * Gives in theory fewer weights
    * But needs weight matrices for both $ATT$ and $MSG$
* Results:
  * Performs as well as PinSAGE
  * Is stable
  * No random walks required
  * Much slower on cpu, comparative on gpu



### Relative temporal encoding through Positional Embeddings

* HGT non conference paper explores temporal encoding through Positional Embeddings
  * By abusing transformers their ability to do positional embedding for sentences (language)



<img src="file:///home/egordm/.config/marktext/images/2022-03-09-12-43-27-image.png" title="" alt="" width="665">



* Cons:
  * Difficult to employ for for clustering (cluster centers) 
  * May get a strong positional embedding bias
  * Relative embedding but with a fixed resolution
    * Resolution needs to be tuned 
    * Can not encode all temporal distances
* Pros:
  * Interpretable
  * Querying possibilities

<img title="" src="file:///home/egordm/.config/marktext/images/2022-03-09-12-46-49-image.png" alt="" width="679">







## What next?

* Finish up temporal learning for heterogenous graphs
* Run the training on all the datasets
* Parameter and network fine tuning
* Experiment with possible integration of temporal encoding through positional embeddings
  * Experiment with possibly identifying influential nodes and building clusters around them
  * This is third step for the algorithm
    * To explain cluster centers
    * Splitting into arbitrary amount of subclusters


