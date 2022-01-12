# Preliminaries

% Goals:
% 
% * Introduce notation and important concepts that we use throughout this proposal
% * Notation is mostly the same as one used in the literature the next chapter is based on

In this section we introduce the notation as well as important concepts that we use throughout this proposal. The notation is mostly the same as one used in the literature the next chapter is based on.



#### Graphs

% * G (V, E)
%   * Attributed edges
%   * Parallel edges

A graph $G$ is a tuple $G = (V, E)$ consisting of $n := |V|$ sequentially numbered nodes $V = \{v_1, ..., v_n\}$ and $m := |E|$ edges. Edges can be directed or undirected, represented as either ordered tuples $(u, v) \in E$ or unordered sets $\{u, v\} \in E$ respectively. Unless stated otherwise we assume that graphs are undirected. An edge $\{u, v\} \in E$ is *incident* to a node $v_i$, if $v_i \in \{u, v\}$. Two nodes $v_i$ and $v_j$ are *adjacent if they are connected by an edge, i.e., $\{v_i, v_j\} \in E$.  The neighbors*  $\mathcal{N}(v_i) = \{v_j|\{v_i, v_j\} \in E\}$ of a node $v_i$ are nodes that are adjacent to it. The *degree* $k_v := |\mathcal{N}(v)|$ of a node $v$ is its number of neighbors. In more mathematical context a graph can represented as an *adjacency matrix* $A$ where $A_{ij}$-th cell is valued $1$ iff edge $\{v_i, v_j\} \in E$. In some cases we also consider *weighted graphs* $G = (V, E, w)$ with weights $w(v_i, v_j) \in \mathbb{R}$. 

#### Multidimensional Networks

% * Parallel edges

Multidimensional network is an edge-labeled extension of multigraphs which allow for multiple edges between same nodes (referred to as *parallel edges*). A multidimensional is defined as $G = (V, E, D)$, where $D$ is a set of dimensions. Each link is a triple $(u, v, d) \in E$ where $u, v \in V$ and $d \in D$.



#### Heterogeneous Graph

A heterogeneous graph $G = (V, E, O)$ extends the notion of a multidimensional network by defining a node type mapping function $\phi: V \rightarrow O_V$ and an edge type mapping function $\psi: E \rightarrow O_E$. Where $O_V$ and $O_E$ denote sets of predefined node and edge types, where $|O_V| + |O_E| > 2$. A *meta-path* $\mathcal{P}$ of length $l$ within a heterogeneous graph is defined in form of $V_{1} \stackrel{R_{1}}{\longrightarrow} V_{2} \stackrel{R_{2}}{\longrightarrow} \cdots \stackrel{R_{l}}{\longrightarrow} V_{l+1}$  (abbreviated as $V_1V_2...V_{l+1}$) which describes a composite relation $\mathcal{R} = R_1 \circ R_2 \circ \cdots \circ R_{l+1}$ between node types $V_1$ till $V_l$. Meta-path based neighbor set $\mathcal{N}^\mathcal{P}(v_i)$ is defned as a set of nodes which connect node $v_i$ via meta-path $\mathcal{P}$.

 

% * Knowledge Graph
% * $KG = (E, R, A, T^R, T^A)$
% * 



#### Temporal Network

A temporal network [@rossettiCommunityDiscoveryDynamic2018] is a graph $G = (V, E, T)$, where $V$ is a set of triplets in form $(v, t_s, t_e)$, with $v$ a node of graph and $t_s, t_e \in T$ respectively being the birth and death timestamps of the corresponding node (with $t_s \leq t_e$); $E$ is a set of quadruplets $(u, v, t_s, t_e)$ with $u, v \in V$ being vertices of the graph, and $t_s, t_e \in T$ respectively being the birth and death timestamps of the corresponding edge (with $t_s \leq t_e$). Networks without edge durations ($t_e = \infty$) are often referred to as *contact sequences* and those with duration as *interval graphs*. Differently, a distinction is made between two types of temporal network, *interaction networks* and *relation networks*. The former model iterations that can repeat as time goes by, while the latter model more stable relationships (friendship, coworker, belonging to same group, etc.).



#### Snapshot Network

A snapshot network $\mathcal{G}_\tau$ [@dakicheTrackingCommunityEvolution2019] is defined by an ordered set $\langle G_1, ... G_\tau \rangle$ of $\tau$ consecutive snapshots, where $G_i = (V_i, E_i)$ represents a graph with only the set of nodes and edges that appear in the interval $(t_i, t_{i+1})$. It is worth noting that a temporal network can be discretized into a snapshot network by partitioning it into a series of snapshots at desired resolution.



#### Complex Network

A complex network is a graph with non-trivial topological features, which do not occur in simple networks such as lattices or random graphs, but often occur in networks representing real systems. A few common features occurring in complex networks include, scale-free networks where the degree distribution follows the power law (implying that degree distribution has no characteristic scale); small-world networks which exhibit small-worldness phenomenen where diameter the network (degree of separation) of is usually small while the clustering coefficient is high; multidimenssional networks; and spatial networks where nodes are embedded in space.





#### Normalized Mutual Information (NMI)

Normalized Mutual Information is a popular measure used to evaluate network partitioning. It is a variant of a common measure in information theory called Mutual Information defined by $I(X; Y) = H(X) - H(X| Y)$ and representing reduction in entropy of variable $X$ by observing the random variable $Y$ or vice versa. In context of network partitioning it therefore quantifies the information shared between two partitions. A lower value represents a higher quality partitioning. NMI is defined as $NMI(X; Y) = \frac{I(X; Y)}{H(X) + H(Y)}$ and ensures the resulting value is within a $[0, 1]$ range, therefore allowing for comparison of different size partitions. 
