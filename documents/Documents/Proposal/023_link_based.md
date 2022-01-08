## Link-based Approaches

% Goals:
% 
% * Describe the intuition of link based approaches
%   * Inter-connection density vs intra connection density
% * Cover the current state of the are
% * Cover different approaches and problems that may arise within the link based approaches
% * Start with community detection
% * Expand by covering different **strategies to tackle tracking instability**

Link-based approaches to (Dynamic) Community Detection rely on connection strength to find communities within the network. The main criteria for communities are the assumed property that intra-group connections are denser than inter-group ones. The networks are partitioned in such a way, that optimizes for a defined measure characterizing this property.

We start this section by covering the fundamentals of link-based community detection by introducing commonly used community quality measures and algorithms for optimizing them. Next, we introduce the link-based DCD problem and the unique challenges that arise as opposed to CD. Then we proceed to cover the current state of the art by describing the related works, their solutions to the said challenges, and possible extensions to the problem.



### Community Detection

% * Talk about basic and common CD techniques
% * Introduce notion of modularity

Different metrics exist quantifying the characteristic of *homophily* over edge strength. The most common metric is Modularity which measures the strength of the division of a network into modules (communities). Its popularity stems from the fact that it is bounded and cheap to compute, though it has other problems such as resolution limit (making detecting smaller communities difficult). Other metrics that can be found in the literature include but are not limited to:

* Conductance: the percentage of edges that cross the cluster border
* Expansion: the number of edges that cross the community border
* Internal Density: the ratio of edges within the cluster with respect to all possible edges
* Cut Ratio and Normalized Cut: the fraction of all possible edges leaving the cluster
* Maximum/Average ODF (out-degree fraction): the maximum/average fraction of nodes’ edges crossing the cluster border

#### Modularity

% * Measures the density of links inside communities compared to links between communities
% * Value: Calculated over a set of nodes
%   * Ranges between [-1/2, 1]
%   * expected number of edges in computed using a configuration model concept
%     * Edges are split into two stubs and each is randomly rewired with any other stub
%     * Based on node degrees pairwise expected number of edges can be computed
%   * It is positive if number of edges within a group exceeds expected number on basis of chance
%   * [Explanations of terms in equations](https://latex.org/forum/viewtopic.php?t=11318)

Modularity directly measures the density of links inside a graph and is therefore computed on communities (sets of nodes) individually by weighing edges based on community similarity (or exact matching). Calculation of modularity is done by aggregating for each community $r$ and for each pair of nodes $vw$ the difference between the expected connectivity $\frac{k_{v} k_{w}}{2 m}$ (amount of edges between the nodes) and the actual connectivity $A_{vw}$ (existence of an edge) given their degrees ($k_v$ and $k_w$). The final result represents the delta connectivity difference by how much the given graph exceeds a random graph as expected connectivity is determined by a random rewiring graph. Because intra-community pairs are weighted lower than inter-community pairs the score can vary. See +@eq:modularity where $S_{vr}$ indicates membership of node $v$ for community $r$ and $m$ represents the total edge count.

$$
Q=\frac{1}{2 m}\sum_{v w}\sum_{r}\left[\overbrace{A_{v w}}^{\text{Connectivity}}-\underbrace{\frac{k_{v} k_{w}}{2 m}}_{\text{Expected Connectivity}}\right] \overbrace{S_{v r} S_{w r}}^{\text{Community Similarity}}
$$ {#eq:modularity}

#### Louvain Method

% * Hierarchical algorithm
%   * Starts with each node assigned to it’s own community
%     * First small communities are found
%       * For each node i change in modularity is calculated for removing i from its own community
%       * And adding it to a neighbor community
%       * Modularity change can be calculated incrementally (local)
%   * Then produces condensed graph by merging communities (their nodes) into a single node
%     * Repeats this process
% * Optimizes for modularity
%   * Is a heuristic algorithm
%     * Since going through all possible assignments maximizing modularity is impractical

Finding an optimal partition of a graph into communities is an NP-hard problem. This is because, while calculating the modularity score can be done in loglinear time, all possible node to community assignments still have to be considered. Therefore heuristic-based methods such as the Louvain method are usually used.

% Feedback: Cite NP-hard CD

Louvain method [@blondelFastUnfoldingCommunities2008] is a heuristic-based hierarchical clustering algorithm. It starts by assigning each node in the graph to its own community. Then it merges these communities by checking for each node the change in modularity score produced by assigning it to a neighbor community (based on the existence of a connection). Once the optimal merges are performed, the resulting communities are grouped into single nodes and the process is repeated.

Since modularity changes can be computed incrementally, the complexity of this method is $O\left(n \log n\right)$. Additionally, due to the flexibility of the modularity measure, it allows detecting communities in graphs with weighted edges.



#### Label Propagation algorithm

% * Algorithm to find communities in graph (very fast)
% * Uses only network structure as guide
% * Doesn’t require any priors (metrics)
% * Intuition:
%   * Single label quickly becomes dominant in a group of densely connected nodes
%   * But these labels have trouble crossing sparsely connected regions
%   * Nodes that end up with same label can be considered part of same community
% * Algorithm:
%   * Initialize each node to their own label
%   * Propagate the labels, per iteration:
%     * Each node updates its label to one that majority of its neighbors belong
%     * Ties are broken deterministically
%   * Stops when convergence is reached, or max iter
% * Preliminary solution can be assigned before run

Another way to sidestep the complexity issue is using the Label Propagation algorithm as it uses the network structure directly to define partitions and doesn't require any priors (quality metrics). The intuition for the Label Propagation algorithm is as follows: When propagating a label through connections in the network, a single label quickly becomes dominant within a group of densely connected nodes, but these labels usually have trouble crossing sparsely connected regions.

The algorithm starts by assigning each node their own label. After that, for each iteration, each node updates its label to the majority label among its neighbors where ties are broken deterministically. The algorithm stops after a fixed amount of iterations or once it has converged. An important feature of this algorithm is that a preliminary solution can be assigned before each run, therefore only updating existing membership assignments.



### Dynamic Community Detection

% * Expand by covering different **strategies to tackle tracking instability**
% * Extension to Community Detection
%   * Adds community tracking to the equation
%   * Instability of community detection methods becomes a problem (tracking relies on their consistency)
% * Community Drift: 
%   * Caused by relying on the previously found communities
%   * Can cause avalanche of wrong detection

Dynamic Community Detection can be seen as an extension to community detection by the addition of the Community Tracking task. Tracking relies on the coherency and stability of found communities to define their evolution through time. The said properties can not be taken for granted and introduce new challenges when designing DCD methods. The main issue is in fact that they are competitive with each other causing a trade-off between community coherency/quality and community temporal stability.

Various strategies dealing with this trade-off are categorized by @rossettiCommunityDiscoveryDynamic2018 and @dakicheTrackingCommunityEvolution2019 where authors reach a consensus over three main groups. In the following sections, we briefly introduce these strategies and describe the current state of the art in similar order.

#### Independent Community Detection and Matching

% * Works in two stages:
%   * CD methods are applied directly to each snapshot (identify stage)
%   * Then the communities are matched between the snapshots (match stage)
% * Advantages:
%   * Use of unmodified CD algorithms (built on top of exisiting work)
%   * Highly parallelizable
% * Disadvantage:
%   * Instability of community detection algorithms (may give **very** different results if network changes)
%   * Difficult to distinguish between instability of algorithm and evolution of the network

Also referred to as the two-stage approach. Works by splitting the DCD task into two stages. The first stage applies CD directly to every snapshot of the network. Followed by the second stage matching the detected communities between the subsequent snapshots.

The advantages of this approach include the fact that it allows for use of mostly unmodified CD algorithms for the first step and that it is highly parallelizable as both detection and matching steps can be applied to each snapshot independently. The main disadvantage is the instability of underlying CD algorithms which may disrupt the community matching process. Many CD methods may give drastically different results in response to slight changes in the network. During the matching, it becomes difficult to distinguish between this algorithm instability and the evolution of the network.



% @wangCommunityEvolutionSocial2008 (core nodes / leader nodes)
% 
% * Circumvents instability issue by studying most stable part of communities (community core nodes)
% * Observations:
%   * The social network scale inflates when it evolves
%   * Members change dramatically and only a small portion exists stably
%     * Therefore only a few can be relied on
% * Introduce algorithm CommTracker
%   * Relies heavily on core nodes
%   * Example: co-authorship community where core nodes represent famous professors
%   * Core Node Detection Algorithm
%     * Each node evaluates centrality of the nodes linked to it
%     * If a node’s weight is higher than it’s neighbors - then its centrality is increased and neighbors decreased
%       * The change value is set as difference is weight
%     * Nodes with non-negative centrality are core nodes
%   * Core-based Algorithm to track communities
%     * Define a set of rules based on presence of core nodes

In @wangCommunityEvolutionSocial2008 the authors circumvent this instability issue by looking at the most stable part of the communities, namely core/leader nodes. In their research, they observe that in various datasets most of the nodes change dramatically while only a small portion of the network persists stably. To exploit this feature, the algorithm CommTracker is introduced which first detects the said core nodes, and then defines rules to both extract communities as well as their evolutional events. The community members are assigned based on their connectivity relative to core nodes.



% @rossettiANGELEfficientEffective2020
% 
% * Detects overlapping communities 
% * Mainly improves computational complexity
% * Use label propagation to extract overlapping fine grained communities
%   * Very fast and has good quality
%   * Runs label propagation for each node on the graph without said node 
%   * Multiple communities are determined by looking at resulting communities of nodes neighbors
%   * These communities are aggregated in a global list
% * The resulting communities are merged into larger ones
%   * By checking if their overlap exceeds a threshold

@rossettiANGELEfficientEffective2020 proposes a way to detect overlapping communities in dynamic networks. A more robust two-phase community detection method (ANGEL) is proposed to ensure the stability of the found communities. The first phase extracts and aggregates local communities by applying Label Propagation on the Ego-Graph (graph excluding a single node) for each node in network. The found communities are biased due to their partial view of the network and are merged in the second step based on their overlap yielding more stable communities with the possibility of overlap. During the matching for each snapshot, a step forward and backward in time is considered where community splits and merges are detected by reusing the matching criteria of the second phase of the CD step.

#### Dependent Community Detection

% Dependent Community Detection / Temporal Trade-Off Communities Discovery
% 
% * Use snapshots to detect communtities
% * To detect communities in current snapshot, rely on communities from the previous
% * Advantage:
%   * Introduces temporal smoothness (fixes the instability problem mostly)
%   * Does not diverge from the usual CD definition (search at each timestep)
% * Disadvantage:
%   * Not parallelizable
%   * Impacts long term coherence of dynamic communtities
%   * Each steps experience substantial drift compared to what a static algorithm would find

Dependent Community Detection strategy works by detecting communities in each snapshot based on the communities found in the previous snapshots. This approach introduces temporal smoothness because the previous solution is reused making the matching process obsolete. Though as part of the described trade-off it can impact the long-term coherence of the dynamic communities. Mainly, because each step introduces a certain amount of error into the results (community drift) which may get amplified within further iterations (error accumulation). Another disadvantage is the fact that the strategy has limited possibilities of parallelization due to its dependent nature.

To lessen the complexity of continuous re-detection of the communities some algorithms process the changes incrementally by limiting the change to a local neighborhood. While this approach has many benefits, it is important to note that these algorithms face a problem where only applying local changes can cause communities to drift toward invalid ones in a global context.

% @heFastAlgorithmCommunity2015 (modified louvain)
% 
% * Modify louvain algorithm to use previous communities
% * They start by detecting communities within the initial graph
% * In subsequent snapshots:
%   * Say that nodes whose edges do not change, stay in the same community
%   * Nodes with changing edges, have to be recomputed
%   * Unchanged nodes maintain their community and are grouped into a single node
%   * Edge weights to changed nodes change proportionally to their connectivity to the community
%   * Louvain method algorithm is run on the newly constructed graph
% * Therefore introducing temporal smoothness and high efficiency

@heFastAlgorithmCommunity2015 introduces an efficient algorithm by modifying the Louvain method. Based on the observation that between consecutive timesteps only a fraction of connections changes and do not affect communities dramatically, they argue that if all community nodes remain unchanged, the community also remains unchanged. With this in mind, they make a distinction between two types of nodes, ones that change the connection in a snapshot transition and ones that do not. The former have to be recomputed, while the latter maintain their community label. The nodes that maintain their community are merged into community nodes with edges to other community nodes and changed nodes weighted proportionally to their real connectivity (amount of edges when ungrouped). This simplified graph is passed to the Louvain method algorithm for community detection. By reusing the community assignments temporal smoothness is maintained and due to the incremental nature of this algorithm, the overall complexity remains low.



% @guoDynamicCommunityDetection2016 (based on distance dynamics)
% 
% * Based on distance dynamics
% * Define distance dynamics in form of a so called Attractor algorithm
%   * Edge weights are initialized using Jaccard distance
%   * Interaction patterns are defined that describe how nodes behave based on their connection to other nodes
%   * Algorithm is run until the weights converge
%   * By removing edges with low distance, communities can be found as connected components
% * Increments can be treated as network disturbance
%   * They can be limited to a certain area by a disturbance factor

@guoDynamicCommunityDetection2016 envision the target network as an adaptive dynamical system, where each node interacts with its neighbors. The interaction will change the distances among nodes, while the distances will affect the interactions. The intuition is that nodes sharing the same community move together, and the nodes in different communities keep far away from each other. This is modeled by defining the so-called Attractor algorithm, which consists of three interaction patterns that describe how node connection strength is influenced by neighboring nodes. The edge weights are initialized using Jaccard distance and the propagation is run until convergence. The communities can be extracted by thresholding on edge weight/distance. Thereafter, all changes are treated as network disturbances. The disturbance can be limited to a certain area using a disturbance factor which defines a bound on the possible propagation.



% @yinMultiobjectiveEvolutionaryClustering2021
% 
% * Look at the problem from an Evolutionary Clustering Perspective
%   * Propose a generic algorithm as in Evolutionary Algorithms
%   * Goal: detect community structure at the current time under guidance of one obtained immediately in the past
%     * Fit observed data well
%     * And keep historical consistency
%   * Combine traditional evolutionary clustering with particle swarm algorithm
% * Solve major drawbacks:
%   * Absence of error correction - which may lead to result-drifting and error accumulation
%   * If initial community structure is not accurate, or the following - this may lead to the “result drift” and “error accumulation”
% * Algorithm:
%   * Use random walks to build a diverse initial population
%   * Search Phase:
%   * Uses custom operators MICO (Communtity Quality) and NBM+ (Community Consistency) to improve global convergence
%     * Cross-over operators - combining multiple individuals (parents)



More recently the @yinMultiobjectiveEvolutionaryClustering2021 has proposed an evolutionary algorithm by looking at the DCD from an Evolutionary Clustering Perspective. They detect community structure at the current time under the guidance of one obtained immediately in the past by simultaneously optimizing for community quality score (modularity) and community similarity between subsequent time steps (NMI). In the methodology, a way is proposed to encode a graph efficiently into a genetic sequence. Additionally, new mutation and crossover operators are proposed which maximize either of the two objectives. By using a local search algorithm, building a diverse initial population, and selecting for dominant candidates the communities maximizing both objectives are obtained.

% Feedback: Define NMI here (instead of in evaluation?) - Or move it into community detection?



#### Simultaneous community detection

% Simultaneous community detection / Cross-Time Community Discovery
% 
% * Doesnt consider independently the different steps of the network evolution
% * Advantage:
%   * Doenst suffer from instability or community drift
%   * More potential to deal with slow evolution and local anomalies
% * Disadvantage:
%   * Not based on usual principle of a unique partition with each timestep
%   * Cant handle real time community detection

The final strategy we consider sidesteps the matching issue by considering all snapshots of the dynamic network at once. This is done by flattening the network in the temporal dimension and coupling edges between the same nodes at different timesteps. These approaches usually don't suffer from instability or community drift. The disadvantages include that the standard principle of an unique partition for each time step can't be applied, as only the combined network is used, therefore limiting the number of possible algorithms. Handling real-time changes to the graph are also usually not considered.

% @muchaCommunityStructureTimeDependent2009
% 
% * Connected identical nodes between different timesteps
% * Then used used louvain method to detect communities

@muchaCommunityStructureTimeDependent2009 adopt a simple yet powerful solution to this problem by connecting identical nodes between different time steps within the unified network. On this network, they apply a modified Louvain method algorithm to extract the communities whose members can be split over different timesteps.



% @ghasemianDetectabilityThresholdsOptimal2016
% 
% * Stochastic block model based approach
% * Define a way to derive a limit to the detectability as function of strength of a community
%   * Some communities are not detectable - probabliity equal to chance
%   * Model is based on prediction of edges that are generated between the timesteps
% * Define a community detection algorithm using Belief propagation equations
%   * To lean marginal probablilities of node labels over time
%   * The use two edge types spatial edges and temporal edges
%   * Similar to spectral clustering

@ghasemianDetectabilityThresholdsOptimal2016 apply stochastic block model-based approach. They make a distinction between two edge types: (i) spatial edges (edges between neighbors) and (ii) temporal edges (edges between nodes in subsequent timesteps). Using this distinction, they define a Belief Propagation equation to learn marginal probabilities of node labels over time. Additionally in their research, they introduce a way to derive a limit to the detectability of communities. This is, because some communities may not be detectable as their probability nears that of random chance.



% TODO: I promised some extensions in the intro. Should I add them?
% 
% * @liuCommunityDetectionMultiPartite2016  - Heterogenous Networks
% * @wanyeTopologyGuidedSamplingFast2021 - Sampling
