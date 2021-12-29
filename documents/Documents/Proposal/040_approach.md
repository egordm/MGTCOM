# Approach

% * Split our task into multiple sections:
%   * Final result will be Framework for DCD in heterogenous graphs
%   * Set of results comparing this algorithm on current state of the art
%   * Ablation tests in-line with research questions providing evidence for our choices
% * First we talk about setup for experimentation and benchmarking
% * Then we talk about strategy for extracting representation learning and graph / data sampling
% * Then we will define components for our objective function
% * Finally we will discuss how community detection results are extracted 

To find an answer to our research questions, the methodology is split into multiple parts. The parts address selection of baselines and evaluation benchmarks, setting up the base of the algorithm by experimentally selecting the representation learning algorithm, building an appropriate objective function and elaborating on DCD result extraction respectively. While split into parts, it is important to note that these tasks have a large overlap and should not be considered in isolation.

The final result of my thesis will consist of the said framework for DCD within dynamic heterogeneous graphs, set of results comparing the algorithm to the current state-of-the-art approaches on various related tasks, and a set of ablation tests providing empirical corroboration for important design choices. 



## Benchmarks and Baselines

% * Implement or set up necessary baseline algorithms:
%   * Static Representation:
%     * @rozemberczkiGEMSECGraphEmbedding2019
%     * @cavallariLearningCommunityEmbedding2017
%     * @jiaCommunityGANCommunityDetection2019
%   * Dynamic link-based:
%     * @rossettiANGELEfficientEffective2020
%   * Dynamic representation:
%     * @wangEvolutionaryAutoencoderDynamic2020
%     * @maCommunityawareDynamicNetwork2020
% * Implement evaluation benchmarks:
%   * These are meant to provide empircal evidence providing answer to the posed research questions
%     * Research questions are of an Explorative and Quantitative nature
%   * Generic:    
%     * Annotated community detection
%     * Quality metric community detection (Modularity)
%   * Task specific: (on public datasets / on own dataset)
%     * Link prediction
%     * Link recommendation

As the research questions posed in +@research-questions are all mostly of a quantitative nature, it is very important to set up appropriate benchmarks in order to provide valid answer. As a direct comparison between methods is not always possible, we define auxiliary task benchmarks testing algorithms on desired properties as well as reuse benchmarks used in previous literature to provide a fair comparison.

### Benchmarks and Evaluation

% RQ 1 (@thm:rq1)
% 
% * Qualitative comparison of temporal and static communities
%   * Link-based:
%     * Annotated: NMI, NF1
%     * Quality Metric: Modularity, Conductance
%   * Representation based:
%     * Clustering measures
%     * Stability measures
%   * Task based:
%     * Content Recommendation
%     * Friend Recommendation
% * Test improvement based on inclusion of content and metadata

To provide an answer for RQ @thm:rq1, the quality of the algorithm on both static and dynamic communities needs to be compared against the benchmarks for various configurations (considering content and/or meta-topological data). As our baselines include both representation- as well as link-based approaches, the benchmarks should cover measures used in both groups. To evaluate the quality of the communities, annotation-based approaches (computing NMI and NF1) and quality metric based evaluation approaches will be employed (See @evaluation). Since our definition of community slightly differs from the literature as it encompasses network external information (content) we will also employ task based evaluation such as recommendation tasks (follower recommendation, hashtag recommendation - depending on the dataset).

Similar evaluation methodology will be employed for RQ @thm:rq3, though the question is of a more exploratory nature concerning modelling of temporal information, and therefore will be conducted as an ablation test with focus on stability measure. 

The research question @thm:rq2 aims to compare the scalability of our approach to the current representation-based approaches. Therefore a rough complexity analysis as well as performance benchmarking (computation time) shall be conducted.

Finally, RQ @thm:rq4 addresses the usability of our dynamic community detection results to other tasks concerning dynamic node representation learning. Here we, make use of defined auxiliary tasks (recommendation and link-prediction) to compare our method against more popular dynamic node representation learning algorithms. 



### Baselines

% * Static Representation:
%   * @rozemberczkiGEMSECGraphEmbedding2019
%   * @cavallariLearningCommunityEmbedding2017
%   * @jiaCommunityGANCommunityDetection2019
% * Dynamic link-based:
%   * @rossettiANGELEfficientEffective2020
% * Dynamic representation:
%   * @wangEvolutionaryAutoencoderDynamic2020
%   * @maCommunityawareDynamicNetwork2020

To give a fair representation of the state-of-the-art the following methods are selected as baselines. The selection is based on their the category of communities they learn, diversification of techniques and competitiveness with the ideas introduced as part of our framework.

#### Static Community Detection:

* Link-based: 
  * @heFastAlgorithmCommunity2015
  * @rossettiANGELEfficientEffective2020
* Representation-based:
  * @rozemberczkiGEMSECGraphEmbedding2019    
  * @cavallariLearningCommunityEmbedding2017
  * @jiaCommunityGANCommunityDetection2019

#### Dynamic Community Detection:

* Link-based:
  * @rossettiANGELEfficientEffective2020
* Representation-based:
  * @wangEvolutionaryAutoencoderDynamic2020
  * @maCommunityawareDynamicNetwork2020

#### Dynamic Representation Learning:

* @nguyenContinuousTimeDynamicNetwork2018
* @wuSageDyNovelSampling2021  
  
  

## Representation learning

% * Core part of the framework as it is responsible for graph sampling and embedding inference
% * Select a Graph Representation Learning technique
% * Select a sampling technique (random walk, convolution)

The core part of the proposed framework is the representation learning algorithm as it is responsible for both random walk sampling of the graph, as well as provides the architecture for the deep neural network used to learn or map the node representation.



## Objective Function

% Build an objective function
% 
% * Homophily (First order + Second order?)
% * Community Cohesion
%   * Sampling based (pairs or motifs)
%   * Similar to AGM
%   * or Modeling the clusters explicitly
% * Temporal Smoothness
% * Community Temporal Smoothness
%   * Sampling based (pairs or motifs)



## Community Detection

% Pick a community detection / grouping algorithm
% 
% * May be by using knn and fixed size groups
% * May be using BIRCH or OPTICS
% * May be using augmentation based method
% * Depending on implementation communities may already be found



## Extensions

% If the timing permits:
% 
% * Explore event detection possibilities in the results
