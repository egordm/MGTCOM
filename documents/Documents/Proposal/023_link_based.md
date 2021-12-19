## Link-based Approaches

% Goals:
% 
% * Describe the intuition of link based approaches
%   * Inter-connection density vs intra connection density
% * Cover the current state of the are
% * Cover different approaches and problems that may arise within the link based approaches
% * Start with community detection
% * Expand by covering different **strategies to tackle tracking instability**

Link-based approaches to (Dynamic) Community Detection rely on connection strength to find communities within the network. The main criteria for communities is the assumed property that intra-group connections are denser than the inter-group ones. The networks are partitioned is such a way, that optimizes for a defined measure characterizing this property. 

We start this section by covering the fundamentals of link-based community detection by introducing commonly used community quality measures and algorithms for optimizing them. Next we introduce link-based DCD problem and the unique challenges that arise as opposed to CD. Then we proceed to covering the current state of the art by describing the related works, their solutions to the said challenges and possible extensions to the problem.



### Community Detection

% * Talk about basic and common CD techniques
% * Introduce notion of modularity

Different metrics exist quantifying the characteristic of *homophily* over edge strength. The most common metric is Modularity which measures the strength of division of a network into modules (communities). It's popularity stems from the fact that it is bounded and cheap to compute, though it has other problems such as resolution limit (making detecting smaller communities difficult). Other metrics that can be found in the literature include but are not limited to:

* Conductance: the percentage of edges that cross the cluster border
* Expansion: the number of edges that cross the community border
* Internal Density: the ratio of edges within the cluster with respect to all possible edges
* Cut Ratio and Normalized Cut: the fraction of all possible edges leaving the cluster
* Maximum/Average ODF: the maximum/average fraction of nodes’ edges crossing the cluster border

#### Modularity

% * Measures the density of links inside communities compared to links between communities
% * Value: Calculated over a set of nodes
%   * Ranges between [-1/2, 1]
%   * expected number of edges in computed using a configuration model concept
%     * Edges are split into two stubs and each is randomly rewired with any other stub
%     * Based on node degrees pairwise expected number of edges can be computed
%   * It is positive if number of edges within a group exceeds expected number on basis of chance

Modularity directly measures the density of links inside a graph and is therefore computed on communities (sets of nodes) individually by weighing edges based on community similarity (or exact matching). Calculation of modularity is done by aggregating for each pair of nodes the difference between the expected connectivity (amount of edges between the nodes) and the actual connectivity (existence of an edge) given their degrees (+@eq:modularity). The final result represents the delta difference by how much the given graph exceeds a random graph as expected connectivity is determined by a random rewiring graph. Because, intra-community pairs are weighted lower than inter-community pairs the score can vary. 

$$
Q=\frac{1}{2 m}\sum_{v w}\sum_{r}\left[\overbrace{A_{v w}}^{\text{Connectivity}}-\underbrace{\frac{k_{v} k_{w}}{2 m}}_{\text{Expected Connectivity}}\right] \overbrace{S_{v r} S_{w r}}^{\text{Community Similarity}}
$$

{#eq:modularity}



#### Louvain Method

% The Louvain method is a popular algorithm to detect communities in large networks. It is a hierarchical clustering algorithm, as it recursively merges communities into single community and further executes modularity clustering on this condensed network.
% 
% @blondelFastUnfoldingCommunities2008
% 
% * Common algorithm for maximizing modularity
% * Hierarchical algorithm
%   * Starts with each node assigned to it’s own community
%   * First small communities are found
%     * For each node i change in modularity is calculated for removing i from its own community
%     * And adding it to a neighbor community
%     * Modularity change can be calculated incrementally (local)
%   * Then produces condensed graph by merging communities (their nodes) into a single node
%   * Repeats this process
% * Optimizes for modularity
%   * Is a heuristic algorithm
%     * Since going through all possible assignments maximizing modularity is impractical
% * TODO: Should I go deeper into calculation?





#### Label Propagation algorithm

% * Algorithm to find communities in graph (very fast)
%   * Uses only network structure as guide
%   * Doesn’t require any priors (metrics)
%   * Intuition:
%     * Single label quickly becomes dominant in a group of densely connected nodes
%     * But these labels have trouble crossing sparsely connected regions
%     * Nodes that end up with same label can be considered part of same community
%   * Algorithm:
%     * Initialize each node to their own label
%     * Propagate the labels, per iteration:
%       * Each node updates its label to one that majority of its neighbors belong
%       * Ties are broken deterministically
%     * Stops when convergence is reached, or max iter
%   * Preliminary solution can be assigned before run



### Dynamic Community Detection

#### Independent Community Detection and Matching

% * (Instant Optimal Communities Discovery)
%   * Works in two stages:
%     * CD methods are applied directly to each snapshot (identify stage)
%     * Then the communities are matched between the snapshots (match stage)
%   * Advantages:
%     * Use of unmodified CD algorithms (built on top of exisiting work)
%     * Highly parallelizable
%   * Disadvantage:
%     * Instability of community detection algorithms (may give **very** different results if network changes)
%     * Difficult to distinguish between instability of algorithm and evolution of the network
% * @wangCommunityEvolutionSocial2008 (core nodes / leader nodes)
%   * Circumvents instability issue by studying most stable part of communities (community core nodes)
%   * Datasets:
%     * enron
%     * imdb
%     * caida
%     * apache
%   * Observations:
%     * The social network scale inflates when it evolves
%     * Members change dramatically and only a small portion exists stably
%       * Therefore only a few can be relied on
%   * Introduce algorithm CommTracker
%     * Relies heavily on core nodes
%     * Example: co-authorship community where core nodes represent famous professors
%     * Core Node Detection Algorithm
%       * Each node evaluates centrality of the nodes linked to it
%       * If a node’s weight is higher than it’s neighbors - then its centrality is increased and neighbors decreased
%         * The change value is set as difference is weight
%       * Nodes with non-negative centrality are core nodes
%     * Core-based Algorithm to track communities
%       * Define a set of rules based on presence of core nodes
%       * To detect the evolu
% * @greeneTrackingEvolutionCommunities2010 (similarity metric based - quality function)
%   * Use quality funciton to quantify similarity between communities
%   * Jaccard Similarity
%   * MOSES: for CD
% * @sunMatrixBasedCommunity2015 ()
%   * Louvain algorithm for CD
%   * Use correlation matrix for community matching
%     * relation between t and t+1
%   * Defined rules to detect evolution events based on matrix
% * @rossettiANGELEfficientEffective2020



#### Dependent Community Detection

% * Dependent Community Detection / Temporal Trade-Off Communities Discovery
%   * Use snapshots to detect communtities
%   * To detect communities in current snapshot, rely on communities from the previous
%   * Advantage:
%     * Introduces temporal smoothness (fixes the instability problem mostly)
%     * Does not diverge from the usual CD definition (search at each timestep)
%   * Disadvantage:
%     * Not parallelizable
%     * Impacts long term coherence of dynamic communtities
%     * Each steps experience substantial drift compared to what a static algorithm would find
% * @heFastAlgorithmCommunity2015 (modified louvain)
%   * Modify louvain algorithm to use previous communities
% * @sunGraphScopeParameterfreeMining2007 (mdl based)
%   * FIrst step: apply MDL method to encode graph with min number of bits
%   * Divide network into multiple sequential segments
%     * Jumps between segments mark evolution over time
%   * Incrementally initialize using previous output
% * @gaoEvolutionaryCommunityDiscovery2016 (leader nodes)
%   * Propose evolutionary CD algorithm based on leader nodes
%   * Each community is a set of followers around a leader
%   * Present an updating strategy with temp info to get initial leader nodes
%   * Leader nodes ensure temporal smoothneess
% * @yinMultiobjectiveEvolutionaryClustering2021
%   * Is a genetic algorithm
%   * Look at the problem from an Evolutionary Clustering Perspective
%     * Evolutionary as in Evolutionary Algorithms
%     * Goal: detect community structure at the current time under guidance of one obtained immediately in the past
%       * Fit observed data well
%       * And keep historical consistency
%   * Solve major drawbacks:
%     * Absence of error correction - which may lead to result-drifting and error accumulation
%     * NP-hardness of modularity based community detection
%     * If initial community structure is not accurate, or the following - this may lead to the “result drift” and “error accumulation”
%   * Propose DYN-MODPSO
%     * Combine traditional evolutionary clustering with particle swarm algorithm
%     * Propose a strategy to detect initial clustering - with insight of future ones
%     * A new way to generate diverse population of individuals
%   * Use previous work DYN-MOGA
%     * Which introduces Multi-Objective optimization
%       * (Trade-off between - Both objectives are competitive)
%       * Accuracy in current time step (CS - Community Score)
%       * Similarity of two communities in consecutive steps (NMI - Similarity of two partitions)
%   * Algorithm:
%     * Initialization Phase:
%       * De redundant random walk to initialize the initial population (with diverse individuals)
%       * Random walk:
%         * Used to approximate probability of two nodes being linked
%         * Then probability is sorted and is split on (based on Q metric optimizing modularity)
%       * Represent as binary string encoding
%     * Search Phase:
%       * Particle swarm optimization - preserving global historically best positions
%         * Already given initial swarms / clusterings
%         * Identifies best positions for different nodes - and uses swarm velocity to interpolate between them
%       * Builds good baseline clusterings
%     * Crossover & Mutation Phase:
%       * Uses custom operators MICO and NBM+ to improve global convergence
%         * Cross-over operators - combining multiple individuals (parents)
%       * Applies specific operators to maximize NMI of CS
%   * Results:
%     * Seems to perform well and be fast?



#### Simultaneous community detection

% * Simultaneous community detection / Cross-Time Community Discovery
%   * Doest consider independently the different steps of the network evolution
%   * Advantage:
%     * Doenst suffer from instability or community drift
%     * More potential to deal with slow evolution and local anomalies
%   * Disadvantage:
%     * Not based on usual principle of a unique partition with each timestep
%     * Cant handle real time community detection
%   * Approaches fall in categories:
%     * Fixed Memberships, Fixed Properties
%       * Communities can change mambers and nodes cant disappear
%       * @aynaudMultiStepCommunityDetection2011
%     * Fixed Membership, Dynamic Properties
%       * Memberships of communties cant change
%       * But they are assigned a profile based on their temporal …
%     * Evolving Membership, Fixed Properties
%       * Allow membership change along time
%       * Mostly SBM approaches
%         * properties can not change over time
%       * @ghasemianDetectabilityThresholdsOptimal2016
%     * Evolving Membership, Evolving Properties
%       * Dont impose any contraints
%       * Edges are added between existing nodes between consequent snapshots
%       * @muchaCommunityStructureTimeDependent2009
%         * Modify louvain



#### Dynamic Community Detection on Temporal Networks (Evolution)

% * Dynamic Community Detection on Temporal Networks / Online Approach
%   * Works directly on temporal graphs
%   * Network consists of series of changes
%   * Algorithms are iterative
%     * Find communties once
%     * Then update them
% * @zakrzewskaTrackingLocalCommunities2016 (leader nodes - rule based)
%   * Dynamic set expandion
%   * Indrementally update community
%     * 1st step: initial community finding
%     * iteration: new nodes are added
%     * Fitness score is used to assign nodes to communtities
% * @xieLabelRankStabilizedLabel2013 (label propagation)
%   * LabelRankT - based on LabelRank (label propagation) algorithm
%   * When network is updated, label of changed nodes are re-initialized
% * @guoDynamicCommunityDetection2016 (based on distance dynamics)
%   * Based on distance dynamics
%   * Increments can be treated as network disturbance
%   * Uses candidate set to pick pick ROI
%     * Changes are made in communties
%     * Distances are updated
