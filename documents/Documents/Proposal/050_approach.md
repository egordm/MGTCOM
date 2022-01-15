# Approach

% * Split our task into multiple sections:
%   * Final result will be Framework for DCD in heterogenous graphs
%   * Set of results comparing this algorithm on current state of the art
%   * Ablation tests in-line with research questions providing evidence for our choices
% * First we talk about setup for experimentation and benchmarking
% * Then we talk about strategy for extracting representation learning and graph / data sampling
% * Then we will define components for our objective function
% * Finally we will discuss how community detection results are extracted 

To find an answer to our research questions, the methodology is split into multiple parts which address the selection of baselines, evaluation benchmark set up, the architecture of the algorithm, building an appropriate objective function, and elaborating on DCD result extraction respectively. While split, it is important to note that these tasks have a large overlap and won't be considered in isolation.

The final result of my thesis will consist of the said framework for DCD within dynamic heterogeneous graphs, a set of results comparing the algorithm to the current state-of-the-art approaches on various related tasks, and a set of ablation tests providing empirical corroboration for important design choices.



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

As the research questions posed in +@research-questions are all mostly of a quantitative nature, it is very important to set up appropriate benchmarks in order to provide a valid answer. As a direct comparison between methods is not always possible, we define auxiliary task benchmarks testing algorithms on desired properties as well as reuse benchmarks used in previous literature to provide a fair comparison.





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

A similar evaluation methodology will be employed for @rqq:rq3, though the question is of a more exploratory nature concerning the modeling of temporal information, and therefore will be conducted as an ablation test with a focus on stability measures.

The @rqq:rq2 aims to compare the scalability of our approach to the current representation-based approaches. Therefore a rough complexity analysis, as well as performance benchmarking (computation time), shall be conducted.

Finally, @rqq:rq4 addresses the usability of our dynamic community detection results to other tasks concerning dynamic node representation learning. Here we, make use of defined auxiliary tasks (recommendation and link-prediction) to compare our method against more popular dynamic node representation learning algorithms.



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
% * Subcomponents:
%   * Graph random walk algorithm (time aware) - random walk or convolution
%   * Representation learning function (shallow or deep)
%   * Feature fusion

The core part of the proposed framework is the representation learning algorithm as it is responsible for both random walk sampling of the graph, as well as provides the architecture for the deep neural network used to learn or map the node representation. The representation learning can be split into three main components. Each of the components respectively has sufficient prior work and has been used in literature. The challenge lays in combining them into an efficient architecture for representation learning capable of maximizing set objective function.



### Graph Sampling

The first component concerns efficient graph sampling, which is a way to enforce learning of desired topological properties from the network. Various ways to sample graphs can be found in the literature. For this work, the choice lies between random walk approaches [@perozziDeepWalkOnlineLearning2014; @groverNode2vecScalableFeature2016; @groverNode2vecScalableFeature2016] which traverse the graph in a depth-first search manner which is known for favoring homophily and convolution-based approaches (@kipfSemiSupervisedClassificationGraph2017a; @hamiltonInductiveRepresentationLearning2018) which sample the graph in a breadth-first search manner favoring structural equivalence. Both methods support heterogeneous graph sampling [@wuAuthor2VecFrameworkGenerating2020; @yingGraphConvolutionalNeural2018; @yangHeterogeneousNetworkRepresentation2020; @dongMetapath2vecScalableRepresentation2017; @wangHeterogeneousGraphAttention2019], and temporally aware sampling [@nguyenContinuousTimeDynamicNetwork2018; @wuSageDyNovelSampling2021; @dasguptaHyTEHyperplanebasedTemporally2018].

% * TODO: BFS random walk is not equivalent to graph convolution?



### Neural Network Architecture

The architecture and training strategy of the underlying neural network matter. Many methods introduced in the previous section already outline an effective architecture suitable for training on the introduced sampling method. Because both data and requirements for our algorithms differ, they can't be used as is. The final architecture should allow for heterogeneous graph samples, temporal-aware samples, and node features.

Similarly, related literature can be used as inspiration as various approaches already utilize content-rich networks [@wuAuthor2VecFrameworkGenerating2020; @yingGraphConvolutionalNeural2018], attention-based mechanisms [@abu-el-haijaWatchYourStep2018; @sankarDynamicGraphRepresentation2019; @wangHeterogeneousGraphAttention2019], and alternative training strategies such (variational or diffusion) auto-encoders, GAN [@liVariationalDiffusionAutoencoders2020; @kipfVariationalGraphAutoEncoders2016].



### Feature Fusion

Many of the presented datasets are feature-rich. While for some feature types using them is as simple as pre-processing the values and passing them to the DNN, some features require additional attention. The natural text features can be aggregated into a single representation vector using pre-trained embeddings [@devlinBERTPretrainingDeep2019; @penningtonGloveGlobalVectors2014], and (large) categorical features may be transformed into network nodes, thus moving the information into topological domain [@chenCatGCNGraphConvolutional2021; @wuTopologicalMachineLearning2020].

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

A very important part of representation-based DCD methods is the objective function. By utilizing node (and community) representation vectors one can optimize the network to maximize a multi-objective function using back-propagation. During the training process, the focus can be shifted between the different objectives. By focusing on defining necessary criteria for dynamic community detection which will give a rough overview of the multi-objective function we will use.

#### Cohesion

The most common definition states that communities are characterized by more dense inter-community connections compared to intra-community density. Representation methods extend this definition by noting that the density of the connections is can be represented by the topological similarity measure of two nodes. Clustering methods further extend this definition by defining similarity on multi-modal embeddings, therefore keeping the definition consistent for attributed networks. Therefore a viable choice for community cohesion would be the Silhouette Coefficient (See +@evaluation)

#### Homophily

To ensure reliable computation of the cohesion measure, the representation vectors need to be accurate. A way to train these representations is by assuming homophily which states that the more two nodes occur in the same context, the more similar they are. This is translated to network problems by using node neighborhood as context. Either using first-order proximity where nodes should occur in their counterpart's context or by utilizing second-order proximity where two nodes are similar if they share the same context. Hybrid approaches exist which optimize for both as they both model different semantics. This idea can also be extended to content-based features and attributed networks, therefore, extending the definition of a community through transitivity when the above definition for cohesion is utilized.

#### Temporal Smoothness

When talking about dynamic networks and communities, temporal smoothness should also be considered. Between subsequent timesteps, the dynamic networks often evolve, but not by a large amount. While individual nodes may change drastically within a single timestep, the communities are seen as more stable structures within the networks. Therefore the evolution of the communities should not exceed to global (network) or individual (node) evolution rate.

In most of the literature, this temporal smoothness is indirectly handled by result matching or reuse of results from previous timesteps. Within representation-based approaches, this property can be quantified and optimized for. A similar approach is employed to keep the embedding space temporally stable while only individual nodes may change.



## Community Detection

% Pick a community detection/grouping algorithm
% 
% * May be by using KNN and fixed-size groups
% * May be using BIRCH or OPTICS
% * May be using augmentation based method
% * Depending on implementation communities may already be found

As the final step of the framework, the viable dynamic communities need to be extracted. This may be done by simultaneously training community embeddings along with the node embeddings [@maCommunityawareDynamicNetwork2020; @limBlackHoleRobustCommunity2016; @wangEvolutionaryAutoencoderDynamic2020], therefore having the advantage that objective function can directly influence the resulting communities. Other approaches instead operate on the resulting embedding space or the augmented graphs to extract the resulting communities using link-based methods such as the Louvain method or density-based clustering algorithms such as K-means, BIRCH [@zhangBIRCHEfficientData1996], or OPTICS [@ankerstOPTICSOrderingPoints1999] yielding the benefit of losing the community count assumption.

In our approach, we plan to focus on direct community optimization, while avoiding hard-coding the model to specific assumptions using spectral clustering-based techniques and soft assignment clustering [@liDivideandconquerBasedLargeScale2021; @maCommunityawareDynamicNetwork2020].

## Extensions

% If the timing permits:
% 
% * Explore event detection possibilities in the results

If the timing of the research project permits, while not the main focus of the research, additional extensions of the algorithms for future work may be explored. These extensions may encompass exploring evolutional event detection within the extracted dynamic communities.
