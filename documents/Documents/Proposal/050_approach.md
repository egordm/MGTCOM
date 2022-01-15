# Approach

% * Split our task into multiple sections:
%   * Final result will be Framework for DCD in heterogeneous graphs
%   * Set of results comparing this algorithm on the current state of the art
%   * Ablation tests in-line with research questions providing evidence for our choices
% * First we talk about setup for experimentation and benchmarking
% * Then we talk about strategy for extracting representation learning and graph/data sampling
% * Then we will define components for our objective function
% * Finally we will discuss how community detection results are extracted 

To answer our research questions, the methodology is split into multiple parts which address the algorithm architecture, construction of the objective function, DCD result extraction, selection of the baselines, and setup of evaluation benchmarks. While split, it is important to note that these tasks have a large overlap and won't be considered in isolation.

The final result of my thesis will consist of the graph representation learning-based framework for DCD within dynamic heterogeneous graphs, a set of results comparing the algorithm to the current state-of-the-art approaches on various related tasks, and a set of ablation tests providing empirical corroboration for important design choices.



## Representation learning

% * Core part of the framework as it is responsible for graph sampling and embedding inference
% * Subcomponents:
%   * Graph random walk algorithm (time aware) - random walk or convolution
%   * Representation learning function (shallow or deep)
%   * Feature fusion

The core part of the proposed framework is the representation learning algorithm as it is responsible for both graph sampling, as well as provides the architecture for the deep neural network (DNN) used to learn the node representation. The representation learning can be split into three main components. Each of the components respectively has sufficient prior work and has been used in literature. The challenge is in combining them into an efficient architecture for representation learning capable of maximizing set objective function.

### Graph Sampling

Graph sampling is a way to enforce the learning of desired topological properties from the network. Various ways to sample graphs can be found in the literature. For both shallow as well as graph representation function learning approaches the trade-off lies between depth-first search (DFS) [@tangLINELargescaleInformation2015; @groverNode2vecScalableFeature2016] and breadth-first search (BFS) [@perozziDeepWalkOnlineLearning2014] based approaches. While the literature has shown that DFS based approaches work better for optimizing homophily [@groverNode2vecScalableFeature2016], our goal is to explore the benefits of both as a hybrid approach within our multi-objective setting. Additionally, we plan on supporting both support heterogeneous graph sampling [@wuAuthor2VecFrameworkGenerating2020; @yingGraphConvolutionalNeural2018; @yangHeterogeneousNetworkRepresentation2020; @dongMetapath2vecScalableRepresentation2017; @wangHeterogeneousGraphAttention2019] and temporally aware sampling [@nguyenContinuousTimeDynamicNetwork2018; @wuSageDyNovelSampling2021; @dasguptaHyTEHyperplanebasedTemporally2018] by adopting extensions described in the literature.

% * TODO: BFS random walk is not equivalent to graph convolution?

### Neural Network Architecture

The architecture and training strategy of the underlying neural network is crucial for a well-performing algorithm. Many methods introduced in the previous section already outline an effective architecture suitable for training on the introduced sampling method. Because both data and requirements for our task differ, the algorithms can not be used out of the box. The final architecture should support heterogeneous graph samples, temporal-aware samples, and node features.

The core of our network will be a representation function-based algorithm [@hamiltonInductiveRepresentationLearning2018; @yingGraphConvolutionalNeural2018] adopted to deal with different node and edge types, therefore generating embeddings for a node's sampled neighborhood in time. Related literature will be used as inspiration to improve model's performance in content-rich networks [@wuAuthor2VecFrameworkGenerating2020; @yingGraphConvolutionalNeural2018], using attention-based mechanisms [@abu-el-haijaWatchYourStep2018; @sankarDynamicGraphRepresentation2019; @wangHeterogeneousGraphAttention2019], and alternative training strategies such (variational or diffusion) auto-encoders, GAN [@liVariationalDiffusionAutoencoders2020; @kipfVariationalGraphAutoEncoders2016].

### Feature Fusion

Many of the real-world datasets are feature-rich. Some features are structured and can be passed to the neural network with minimal pre-processing. Some feature types require additional attention. Natural (unstructured) text features can be aggregated into a single representation vector using pre-trained embeddings [@devlinBERTPretrainingDeep2019; @penningtonGloveGlobalVectors2014], and (large) categorical features may be transformed into network nodes, thus moving the information into the topological domain [@chenCatGCNGraphConvolutional2021; @wuTopologicalMachineLearning2020].

## Objective Function

% Build an objective function
% 
% * Homophily (First order + Second-order?)
% * Community Cohesion
%   * Sampling-based (pairs or motifs)
%   * Similar to AGM
%   * or Modeling the clusters explicitly
% * Temporal Smoothness
% * Community Temporal Smoothness
%   * Sampling-based (pairs or motifs)

A crucial part of representation-based DCD methods is the objective function. By utilizing node (and community) representation vectors one can optimize the network to maximize a differentiable multi-objective function using back-propagation. During the training process, the focus can be shifted between the different objectives. By focusing on defining necessary criteria for dynamic community detection which will give a specification for our multi-objective function.

#### Cohesion

The most common definition states, that communities are characterized by more dense inter-community connections compared to intra-community density. Representation methods extend this definition by noting that the density of the connections is can be represented by the topological similarity measure of two nodes. Clustering methods further extend this definition by defining similarity on multi-modal embeddings, therefore keeping the definition consistent for feature-rich networks. A viable choice for community cohesion measure would be the Silhouette Coefficient (See +@evaluation)

#### Homophily

To ensure reliable computation of the cohesion measure, the representation vectors need to be accurate. A way to train these representations is by assuming homophily which states that the more two nodes occur in the same context, the more similar they are. This is translated to graph representation learning problems by using node neighborhood as context. Either using first-order proximity where nodes should occur in their counterpart's context or by utilizing second-order proximity where two nodes are similar if they share the same context (through common nodes). Hybrid approaches exist which optimize for both as they both model different semantics. Similarly, this idea can also be extended to feature aware embeddings, therefore, extending the definition of a community through transitivity when the above definition for cohesion is utilized. Homophily is usually measured by the distance of connected nodes their embeddings within a similarity space (euclidean, cosine, etc.). 

#### Temporal Smoothness

When talking about dynamic networks and communities, temporal smoothness should also be considered. Between subsequent timesteps, the dynamic networks often evolve, but not by a large amount. While individual nodes may change drastically within a single timestep, the communities are seen as more stable structures within the networks. Therefore the evolution of the communities should not exceed to global (network) evolution rate.

In most of the literature, this temporal smoothness is indirectly handled by result matching or reuse of results from previous timesteps. Within representation-based approaches, this property can be quantified and optimized for using cross-times measures. A similar approach is employed to keep the embedding space temporally stable while only individual nodes may change.

## Community Detection

% Pick a community detection/grouping algorithm
% 
% * Maybe by using KNN and fixed-size groups
% * May be using BIRCH or OPTICS
% * May be using augmentation based method
% * Depending on implementation communities may already be found

In the final step of the framework, dynamic communities need to be identified. This may be done by simultaneously training community embeddings along with the node embeddings [@maCommunityawareDynamicNetwork2020; @limBlackHoleRobustCommunity2016; @wangEvolutionaryAutoencoderDynamic2020], therefore having the advantage that objective function can directly influence the resulting communities. Other approaches instead operate on the resulting embedding space or the augmented graphs to extract the resulting communities using link-based methods such as the Louvain method or density-based clustering algorithms such as K-means, BIRCH [@zhangBIRCHEfficientData1996], or OPTICS [@ankerstOPTICSOrderingPoints1999] yielding the benefit of losing the community count assumption.

In our approach, we plan to focus on direct community optimization, while avoiding hard-coding the model to specific assumptions using spectral clustering-based techniques and soft assignment clustering [@liDivideandconquerBasedLargeScale2021; @maCommunityawareDynamicNetwork2020].





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

As the research questions posed in +@research-questions are all mostly of a quantitative nature, it is very important to set up appropriate benchmarks to provide a valid answer. As a direct comparison between methods is not always possible, we define auxiliary task benchmarks for testing algorithms on desired properties as well as reuse benchmarks used in previous literature to provide a fair comparison.



### Benchmarks and Evaluation

% RQ 1 (@rqq:rq1)
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

To provide an answer for @rqq:rq1, the quality of the algorithm on both static and dynamic communities needs to be compared against the benchmarks for various configurations (considering the content and/or meta-topological data). As our baselines include both representation- as well as link-based approaches, the benchmarks should cover measures used in both groups. To evaluate the quality of the communities, annotation-based approaches (computing NMI and NF1) and quality metric-based evaluation approaches will be employed (See +@evaluation). Since our definition of community slightly differs from the literature as it encompasses network external information (content) we will also employ task-based evaluation such as recommendation tasks (follower recommendation, hashtag recommendation - depending on the dataset).

The @rqq:rq3 is of a more exploratory nature concerning the modeling of temporal information. It aims to determine whether having the ability to track the communities through time yields better results in practice, as opposed to having communities incorporate their temporal information implicitly in their definition. This evaluation is conducted as an ablation test and similarly focuses on quality measures and task-based evaluation.

The @rqq:rq2 aims to compare the scalability of our approach to the current representation-based approaches. Therefore a rough complexity analysis, as well as performance benchmarking (computation time), shall be conducted.

Finally, @rqq:rq4 addresses the usability of our dynamic community detection results to other tasks concerning dynamic node representation learning. Here we, make use of defined auxiliary tasks (recommendation and link-prediction) to compare our method against other dynamic node representation learning algorithms.

For benchmarking the datasets of different scales and properties are chosen (+@tbl:benchmarkdatasets). The synthetic dataset generation method introduced in @greeneTrackingEvolutionCommunities2010 will be used to create additional networks with ground-truth communities.



| Dataset                                                                                                | Nodes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Edges&nbsp; | Temporal | Annotated &nbsp; |
| ------------------------------------------------------------------------------------------------------ | ----------------------------------------- | ----------- | -------- | ---------------- |
| [Zachary karate club](http://konect.cc/networks/ucidata-zachary/)                                      | 34                                        | 78          | N        | Y                |
| [Football](https://networkrepository.com/misc-football.php)                                            | 115                                       | 613         | N        | N                |
| [Star Wars Social](https://www.kaggle.com/ruchi798/star-wars)                                          | 113                                       | 1599        | Y        | N                |
| [Enron](https://www.kaggle.com/wcukierski/enron-email-dataset/)                                        | 605K                                      | 4.1M        | Y        | Y                |
| [IMDB 5000](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)                               | 16K                                       | 52K         | Y        | N                |
| [DBLP-HCN](https://data.mendeley.com/datasets/t4xmpbrr6v/1)                                            | 11K                                       | 16K         | Y        | Y                |
| [DBLP-V1](https://www.aminer.org/citation)                                                             | 1.2M                                      | 2.4M        | Y        | Y                |
| [DBLP-V3](https://www.aminer.org/citation)                                                             | 2.7M                                      | 8.2M        | Y        | Y                |
| [sx-mathoverflow](https://snap.stanford.edu/data/sx-mathoverflow.html)                                 | 24K                                       | 506K        | Y        | N                |
| [sx-superuser](https://snap.stanford.edu/data/sx-superuser.html)                                       | 194K                                      | 1.4M        | Y        | N                |
| [Eu-core network](https://snap.stanford.edu/data/email-Eu-core.html)                                   | 1005                                      | 25K         | N        | Y                |
| [com-Youtube](https://snap.stanford.edu/data/com-Youtube.html)                                         | 1.1M                                      | 298K        | N        | Y                |
| [116th House of Representatives](https://www.kaggle.com/aavigan/house-of-representatives-congress-116) | 6249                                      | 12K         | N        | N                |
| [social-distancing-student]()                                                                          | 93K                                       | 3.7M        | Y        | N                |

Table: Overview of the datasets used for evaluation. All the datasets will be used for the quality measure-based as well as auxiliary task-based evaluation. Annotated column indicates whether a dataset is eligible for annotation-based evaluation, ie. it contains ground-truth communities. {#tbl:benchmarkdatasets}



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

To give a fair representation of the state-of-the-art the following methods are selected as baselines. The selection is based on the category of communities they learn, diversification of techniques, and competitiveness with the ideas introduced as part of our framework.

#### Static Community Detection

For @rqq:rq1 we will evaluate our algorithm against baselines in @tbl:baselinescd. We use both static as well as dynamic algorithms as baselines to identify the benefit of dynamic community detection over static communities.

| Reference                                | Dynamic | Method               |
| ---------------------------------------- | ------- | -------------------- |
| @heFastAlgorithmCommunity2015            | N       | Link-based           |
| @blondelFastUnfoldingCommunities2008     | N       | Link-based           |
| @rozemberczkiGEMSECGraphEmbedding2019    | N       | Representation-based |
| @cavallariLearningCommunityEmbedding2017 | N       | Representation-based |
| @jiaCommunityGANCommunityDetection2019   | N       | Representation-based |
| @rossettiANGELEfficientEffective2020     | Y       | Link-based           |
| @wangDynamicCommunityDetection2017       | Y       | Link-based           |
| @greeneTrackingEvolutionCommunities2010  | Y       | Link-based           |
| @wangEvolutionaryAutoencoderDynamic2020  | Y       | Representation-based |
| @maCommunityawareDynamicNetwork2020      | Y       | Representation-based |

Table: List of community detection methods we will use as baselines. The "dynamic" column indicates whether the algorithm is capable of detecting dynamic communities, and the "method" column indicates whether it is a link-based or representation-based algorithm. {#tbl:baselinescd}

#### Dynamic Representation Learning

To answer @rqq:rq3 we will evaluate the algorithm against other dynamic representation learning algorithms to verify that learned node embeddings are still usable for node-level predictions tasks as well as community-level tasks. 

* @nguyenContinuousTimeDynamicNetwork2018
* @wuSageDyNovelSampling2021
* @parejaEvolveGCNEvolvingGraph2020
  
  

## Extensions

% If the timing permits:
% 
% * Explore event detection possibilities in the results

If the timing of the research project permits, while not the main focus of the research, additional extensions of the algorithms for future work may be explored. These extensions may encompass exploring evolutional event detection within the extracted dynamic communities.
