---
type: paper
title: "Community Detection through Representation learning in Evolving Heterogenous Networks"
subtitle: "A Master’s Thesis proposal"
author:
  name: "Egor Dmitriev"
  institute: "Utrecht University"
  country: "The Netherlands"
  email: "e.dmitriev@students.uu.nl"
topic: "dynamic networks, community detection"
bibliography: ../refs.bib
toc: true
abstract: |
  | Recent developments in big data and graph representation learning have allowed researchers to make breakthroughs in social network analysis and the identification of communities. While opening a lot of research opportunities, such approaches are highly limited to snapshots of rapidly evolving social networks. This, in fact, is a great simplification of the real-world situation which is always evolving and expanding by the user and/or machine interactions.
  |
  | Relying on novel research of dynamic graph representation learning, the goal of my thesis project is to build a framework for community detection and representation in evolving heterogeneous networks. To verify the merit of the proposed framework, it will be evaluated against baselines on static heterogeneous graphs, and analyzed against gathered twitter dataset on covid measures.

---



[TOC]



# Introduction and Backgrounds

Social Network Analysis (SNA) is a huge part of the Network Science field and is concerned with the process of investigating social structures that occur in real-world using Network and Graph Theory. These social structures usually include social media networks, economic transaction networks, knowledge networks and disease transmission networks.
One main issue to address while studying this type of real-world events lies in identification of meaningful substructures hidden within the overall complex system. The SNA is therefore applied to extract patterns from the data usually in form of information flow,  identification of high throughput nodes and paths, and discovery of communities and clusters. In this thesis we are going to focus on the problem of community discovery.

This thesis proposal is structured as follows: in this sections we are going to introduce basic concepts and challenges of Dynamic Community Detection. In +@research-questions we will describe the problem we are trying to solve as well as formulate the research questions. In +@literature-review a brief literature survey is conducted on identifying current state of the art and approaches to Community Detections. In +@approach we will elaborate on our methodology for solving the posed problem and answering the research questions. Finally, in +@planning the concrete planning for th research project is laid out.

## Community Detection

Problem of partitioning a complex network into *communities* which represent groups of individuals with high interaction density while individuals from different communities have comparatively low interaction density is known as Community Discovery (CD). CD is a task of fundamental importance within CD as it discloses deeper properties of networks. It provides insight into networks’ internal structure and its organizational principles.

Many useful applications of CD have been studied by researchers including identification of criminal groups [@sarvariConstructingAnalyzingCriminal2014], social bot detection  [@karatasReviewSocialBot2017], targeted marketing [@mosadeghUsingSocialNetwork2011], and public health / disease control [@salatheDynamicsControlDiseases2010].

With the explosion of human- and machine-generated data, often collected by social platforms, more datasets are emerging having rich temporal information that can be studied. CD operates only on static networks. Meaning that their temporal dimension is often omitted, which often does not yield a good representation of the real-world where networks constantly evolve. Such networks are often referred to as dynamic networks as their components such as nodes and edges may appear and fade from existence. Accordingly community detection on such dynamic networks is called Dynamic Community Detection (DCD).

DCD algorithms, by incorporating additional temporal data are often able to both outperform their counterpart CD algorithms @faniUserCommunityDetection2020, as well as providing additional information about communities for analysis [@pallaQuantifyingSocialGroup2007]. This additional information comes in form of community events such as (birth, growth, split, merging, and death)  or in form of ability to track membership of certain individuals over time.

## Challenges in Community Detection

DCD is seen as the hardest problem within Social Network Analysis. Reason for this is mainly because DCD, unlike CD, also involves tracking the found communities over time. This tracking relies on consistency of the detected communities as usually slight changes to the network may cause a different community membership assignment.

Additionally, increasing richness of the data is not only limited to temporal data. The real-world data often connects entities of different modalities together. This multi-modality occurs through the fact that the entities and relations themselves may be of different kinds (meta topology-based features). For example users, topics and user-produced documents in a social network, or vehicles and landmarks in a traffic network. Another example of multi-modality in networks comes in form of node and relation features (content-based features). These features may come in form of structured (numerical, categorical or vector data) or unstructured data such as images and text. It is of high importance to explore this multi-modal data as it may not always be possible to explain the formation of communities using network structural information alone.

As noted earlier, meta topological features may be used to differentiate between different kind of nodes or edges to encode additional information. TODO: talk about appearance and disappearance of nodes, asymmetric edges, etc

Finally, a more common issue is that there is no common definition for a community structure. Within networks it is usually described in terms of membership assignment, while in more content-based settings communities are described in terms of distributions over topics which usually represent interest areas. The first definition only accounts for disjoint communities, while second is too vague as there may also be overlapping and hierarchical communities.

* In contrast to k-means clustering tasks, the amount of communities is unknown

# Literature review

The problem of dynamic community detection was noticed quite early on in within the SNA community and a considerable amount of research have been made in order to provide a comprehensive analysis of the network. While the said research was mostly focused on discovery of communities using topologically-based features and node connectivity, the covered methods did research the limitations and challenges posed by a temporal context. 

In the recent years, significant developments have been made in the space of deep learning. Mainly in the development of new deep learning methods capable of learning graph-structured data  [@bronsteinGeometricDeepLearning2017; @hamiltonRepresentationLearningGraphs2018; @kipfSemiSupervisedClassificationGraph2017] which is fundamental for SNA. Various of problems within the field have been revisited, including the community detection problems. The approaches have been expanded by incorporation of more complex features, solving the problems concerning multi-modality and introduction of unsupervised learning. 

Despite this resurgence, the DCD problem has received little attention. Though a few efforts have been made to incorporate the deep learning methods by introducing content-based similarity dynamic and usage of graph representation based CD algorithms within a temporal context, the current state of the art leaves a lot to be desired. 

We structure this literature review as follows: first we describe the various interpretations of the community structure @community-structures, explore the current datasets and evaluation methods used for benchmarking of the current DCD methods @evaluation-methods. Then, we dive in the current state of the art works on DCD by discussing both “classic” methods and novel deep learning based methods +@datasets. Finally, we discuss the current advances within the graph representation learning [] and community detection [] approaches.

## Community Structures

Communities in real-world networks can be of different kinds: disjoint (think of students belonging to different educational institutions), overlapping (person having membership in different social groups) and hierarchical (components of a car). One of the main reasons behind the complexity of CD is the unclear definition what a community actually is.

### Link-based Perspective

The *link-based* (referred to as classic) community detection methods intuitively describe communities as groups of nodes within a graph, such that the intra-group connections are denser than the intergroup ones. This definition is primarily based on the *homophily* principle, which refers to the assumption that similar individuals are those that are densely connected together. Therefore, these kind of methods look for sub-graph structures such as cliques and components that identify connectedness within the graph structure to represent the communities. 

Unfortunately, in most cases link-based methods fall short to identity communities of similar individuals. This is mainly due to two facts: (i) many similar individuals in a social network are not explicitly connected together, (ii) an explicit connection does not necessarily indicate similarity, but may explained by sociological processes such as conformity, friendship or kinship [@diehlRelationshipIdentificationSocial2007; @faniUserCommunityDetection2020].

A more general definition is introduced in [@cosciaClassificationCommunityDiscovery2011] to create an underlying concept generalizing all variants found in the literature. In link-based methods, a direct connection is considered as a particular and very important kind of action, while newer methods also consider content or interest overlap.

> *Definition (Community). A community in a complex network is a set of entities that share some closely correlated sets of actions with the other entities of the community.* 

### Representation-based Perspective

Representation-based approach stems from the field of computation linguistics which relies heavily on the notion of *distributional semantics* which states that words that occur in similar contexts are semantically similar. Therefore the word representations are learned as dense low dimensional representation vectors (embeddings) of a word in a latent similarity space by predicting words based on their context or vice versa [@mikolovEfficientEstimationWord2013; @penningtonGloveGlobalVectors2014]. Using the learned representations similarity, clustering and other analytical metrics can be computed.

Success of these representation learning approaches has spread much farther than just linguistics and can be applied to graph representation learning. Methods such as deepwalk [@perozziDeepWalkOnlineLearning2014], LINE [@tangLINELargescaleInformation2015] and node2vec [@groverNode2vecScalableFeature2016] use random walks to sample the neighborhood/context in a graph (analogous to sentences in linguistic methods) and output vector representations (embeddings) that maximize the likelihood of preserving topological structure of nodes within the graph. 

Whereas previously the structural information features of graph entities had to be hand engineered, these new approaches are data driven, save a lot of time labeling the data, and yield superior feature / representation vectors. The methods can be trained to optimize for *homophily* on label prediction or in an unsupervised manner on link prediction tasks.

Newer approaches introduce possibility for fusion of different data types. GraphSAGE [@hamiltonInductiveRepresentationLearning2018] and Author2Vec [@wuAuthor2VecFrameworkGenerating2020] introduce methodology to use node and edge features during representation learning process. Other approaches explore ways to leverage heterogeneous information present within the network by using *metapath* based random walks (path defined by a series of node/link types) [@dongMetapath2vecScalableRepresentation2017] or by representing and learning relations as translations within the embedding space [@bordesTranslatingEmbeddingsModeling2013]. In @nguyenContinuousTimeDynamicNetwork2018 the authors introduce a way to encode temporal information by adding chronological order constraints to various random walk algorithms. Other relevant advancements within the field include Graph Convolutional Networks (GCN) [@kipfSemiSupervisedClassificationGraph2017a] and (Variational) Graph Auto-Encoders (GAE) [@kipfVariationalGraphAutoEncoders2016] which present more effective ways to summarize and represent larger topological neighborhoods or whole networks.

Various works have emerged exploring community detection using the representation learning approach. In @cavallariLearningCommunityEmbedding2017 the authors define a community as a distribution over the vector representation (embedding) space of the network (which encodes both content-based as well as topological information).  Here community detection and node representation are jointly solved by defining a unified objective and alternating their optimization steps. @faniUserCommunityDetection2020 redefine node connection proximity based on learned multi-modal embedding vectors incorporating both temporal social content as well as social network neighborhood information. As *homophiliy* is optimized, more valuable communities are found within the resulting network.

* todo: this already talks about used approaches. 
  * Better elaborate on them more and move them into approaches section
  * Move this section into a larger section
    * split and elaborate on CD methods
  

### Dynamic Community

Similarly to how communities can be found in static networks, dynamic communities extends this definition by utilizing the temporal dimension to define its life cycle/evolution over a dynamic network. A dynamic community is defined as a collection of communities and set of transformations on thesis communities over time.

This persistence across time of communities subjected to progressive changes is an important problem to tackle. Since as noted by [@rossettiCommunityDiscoveryDynamic2018] it can be compared to the famous “the ship of Theseus” paradox. Because (verbatim), *deciding if an element composed of several entities at a given instant is the same or not as another one composed of some—or even none—of such entities at a later point in time is necessarily arbitrary and cannot be answered unambiguously*.

Most of the works agree on two atomic transformations on the communities, including node/edge appearance and vanishing. While some such as [@pallaQuantifyingSocialGroup2007; @asurEventbasedFrameworkCharacterizing2009, @cazabetUsingDynamicCommunity2012] define a more extensive set of transformations (also referred to as events) which may be more interesting for analytical purposes:

* Birth, when a new community emerges at a given time.
* Death, when a community disappears. All nodes belonging to this community lose their membership.
* Growth, when a community acquires some new members (nodes).
* Contraction, when a community loses some of its members.
* Merging, when several communities merge to form a new community.
* Splitting, when a community is divided into several new ones.
* Resurgence, when a community disappears for a period and reappears.

These events / transformations are often not explicitly used during the definition and/or representation of dynamic communities. Nevertheless, most of the methods covered discussed in the following sections do defined a way in their algorithm to extract such event from the resulting data.

Finally, is important to note that dynamic networks can differ in representation. They can be represented as either a time-series of static networks (also referred to as snapshots), or as a real time stream of edges (referred to as temporal networks). Though, it should be noted that within the context of dynamic community detection they can be seen as equivalent as the conversion between the two representations can be done in a lossless way.

## Evaluation methods {#evaluation-methods}

As described in the previous section, the definition for both community and dynamic community may be quite ambiguous. In this section we will cover how detection and tracking results can be evaluated in a lesser ambiguous setting to compare various approaches. To disambiguate the process a little, during evaluation, the resemblance/detection and matching/tracking tasks are evaluated separately.

### Evaluation of Link-based methods

#### Annotation Based

Evaluation of detected (dynamic) communities becomes much easier when the *ground truth communities* are provided. The evaluation is then done by comparing the difference between the produced communities and the effective ones. To perform this comparison, information theory based metric Normalized Mutual Information (NMI) is used which converts community sets to bit-strings and quantifies the “amount of information” can be obtained about one community by observing the other [@lancichinettiDetectingOverlappingHierarchical2009]. 

A possible drawback of this measure is that its complexity is quadratic in terms of identified communities. In [@rossettiNovelApproachEvaluate2016] alternative measure (NF1) with linear complexity is introduced which similarly to F1 score uses the trade-off between precision and recall (of the average of harmonic means) of the matched communities. In the follow-up work [@rossettiANGELEfficientEffective2020] the authors describe a way to apply this measure within the context of DCD by calculating this score for all the snapshots and aggregating the results into one single measure.

In real-world there are usually no ground truth communities. Therefore this approach is usually applied on synthetic datasets where the communities and their dynamicity is sampled from a distribution. Alternative approach some papers take is by defining ground truth communities using the metadata and node attributes present within the datasets. Some datasets may include annotated communities, but this is not common within DCD datasets.

#### Metric Based

Another way to evaluate and compare different CD algorithms without knowing ground truth communities is using a quality function. Modularity is the most widely used measure [@newmanFastAlgorithmDetecting2004], since it measures the strength of division of a network into modules. Networks with high modularity have dense connections between the nodes within the modules, and sparse connections between nodes in different modules. Other measures are used as well including:

* Conductance:  the percentage of edges that cross the cluster border
* Expansion:  the number of edges that cross the community border
* Internal Density: the ratio of edges within the cluster with respect to all possible edges
* Cut Ratio and Normalized Cut: the fraction of all possible edges leaving the cluster
* Maximum/Average ODF:  the maximum/average fraction of nodes’ edges crossing the cluster border

#### Alternative Measures

In [@peelGroundTruthMetadata] the authors criticize these evaluation approaches by proving that they introduce severe theoretical and practical problems. For one, they prove the no free lunch theorem for CD, ie. they prove that algorithmic biases that improve performance on one class of networks must reduce performance on others. Therefore, there can be no algorithm that is optimal for all possible community detection tasks, as quality of communities may differ by the optimized metrics. Additionally, they demonstrate that when a CD algorithm fails, the poor performance is indistinguishable from any of the three alternative possibilities: (i) the metadata is irrelevant to the network structure, (ii) the metadata and communities capture different aspects of network structure, (iii) the network itself lacks structure. Therefore, which community is optimal should depend on it’s subsequent use cases and not a single measure.

* Todo: lead into “subsequent” use cases and representation based methods

### Evaluation of Representation methods

* Explore Recommendation / Link prediction task

* @faniUserCommunityDetection2020
  * Compare against Static Content Based CD Algorithms
  * Content Based Community Detection
  * Compare against Static Link Based CD Algorithms
  * Compare against Multimodal Based CD Algorithms
  * Problems:
    * Absence of ground truth communities
    * Modularity cant be used - based  on explicit links betwene users (structural)
      * Doesnt account for content at all
  * Solutions: Application level evaluation 
    * A user community detection method is considered to have better quality iff its output communities improve an underlying application
    * **News recommendation** (in time)
      * Curate dataset of news articles mentioned by users (user mention means user interest)
      * Methodology: 
        * Detect communities and assign them a topic of interest at a time
        * Topic is average of user interests
        * All news articles are ranked based on their similarity with the overall topic (in time)
        * Each member in community is recommended the ranked list 
      * Metrics: (stadard info retreval metrics)
        * Precision at rank $k$ ($P_k$)
          * $\mathrm{P}_{k}=\frac{1}{|\mathrm{U}|} \sum_{u \in \mathbb{U}} \frac{t p_{u}}{k}$
          * $u$ is user
        * Mean Reciprocal Rank (MRR)
          * $\mathrm{MRR}=\frac{1}{|\mathbb{U}|} \sum_{u \in \mathbb{U}} \frac{1}{\operatorname{rank}_{u}}$
          * First position correct result occurs in list
        * Success at rank $k$ ($S_k$)
          * Probability that at least one correct item is within a top-k list
          * $\mathrm{S}_{k}=\frac{1}{|\mathbb{U}|} \sum_{u \in \mathcal{U}}\left(\operatorname{rank}_{u} \leq k\right)$
          * 
    * **User Prediction**
      * Goal: Predict which users posted a news article $a$ at time $t$
      * Methodology:
        * Find closest community to the article in terms of interest at time $t$ (cosine sim)
        * Members of community are predicted users
      * Same reasoning as news prediction
      * Metrics (classificiation metrics)
        * Precision, Recall, F-measure
  
* @wangVehicleTrajectoryClustering2020

  * Use taxi dataset with license plates
  * Compare to other deep GNN - they only learn static representation
  * Metrics:
    * **Silhouette Coefficient** (SC) - range [-1, 1]
      * $S(i)=\frac{b(i)-a(i)}{\max \{a(i), b(i)\}}$
      * $a$ avg distance between node and neighbors in cluster
      * $b$ is min val of avg distances between node and other clusters
    * **Davies-Bouldin Index** (DBI)
      * $D B I=\frac{1}{N} \sum_{i=1}^{N} \max _{j \neq i}\left(\frac{\overline{S_{i}}+\overline{S_{j}}}{\left\|w_{i}-w_{j}\right\|_{2}}\right)$
      * $\bar{S_i}$: avg distance of nodes in cluster $i$ to centroid of cluster $i$
      * $w_i$ is the centroid of cluster $w_i$
      * It is the ratio of the sum of the average distance to the distance between the centers of mass of the two clusters
      * The closer the clustering result is with the inner cluster, and the farther the different clusters, the better the result
    * **Calinski-Harabaz Index** (CHI): Ratio of the between-cluster variance and within-cluster variance
      * $C H I=\frac{\operatorname{tr}\left(B_{k}\right)}{\operatorname{tr}\left(W_{k}\right)} \frac{m-k}{k-1}$
      * $m$ number of nodes, $k$ number of clusters, 
      * $B_k$ is covariance matrix between the clusters
      * $W_k$ is covariance matrix between the data in the cluster
      * $tr$ is trace of the matrix
  
* @maCommunityawareDynamicNetwork2020

  * Use both synthetic and real world datasets
  * Use not perse community detection baselines
  * Define auxilary helper tasks in context of *community aware* **Deep Network Embedding**:
    * **Network Reconstruction**: Evaluates model on ability of reconstructing link structures of the network
      * Average reconstruction precision is measured
      * This is done for each timestamp
      * For each node, the nearest embedding neighbors are used as predicted links
    * **Link Prediction**:
      * Prediction of existence of links between nodes in the next timestamps
      * Based on representation in the current timestamp
    * **Network Stabilization**: evaluates preformance of DNE on stabilization of embedding
      * dynamic network should have similar evolutionary patterns in both the learned low-dimensional representation and the
        network representation over time
      * evaluates the evolution ratio of the low-dimensional node representations to the network representations at $a$ th timestamp
      * $p_{s}^{a}=\frac{\left(\left\|\mathrm{H}^{a+1}-\mathrm{H}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}^{a+1}-\mathrm{A}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}^{a}\right\|_{2}^{2}} .$
    * **Community Stabilization**:  evaluates stability of communities in dynamic networks on the embedded low-dimensional representations
      * Evaluates communty evolution ratio to network representation evolution between subsequent timestamps
      * Lower values point to more stable communities and are better
      * $p_{c}^{a}=\sum_{k=1}^{q}\left(\frac{\left(\left\|\mathrm{H}_{c_{k}}^{a+1}-\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}_{c_{k}}^{a+1}-\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}}\right) / q$

  * Network is first fine-tuned on each of the tasks
  * Note: the evalution is at graph level since their methods are spectral GAE based

* @mrabahRethinkingGraphAutoEncoder2021

  * Accuracy:
  * NMI
  * ARI:
  
* @huangInformationFusionOriented2022
  * Based on link prediction or friend recommendation
  * Precision
  * Recall
  * F-score
  * normalized discounted cumulative gain (nDCG)
  * mean reciprocal rank (MRR)
  
* 

## Datasets

### Synthetic

| Paper                                              | Description                                                  |
| -------------------------------------------------- | ------------------------------------------------------------ |
| @lancichinettiBenchmarkGraphsTesting2008           | Static networks (widely used)                                |
| @greeneTrackingEvolutionCommunities2010            | Generate Graphs based on Modularity measure                  |
| @granellBenchmarkModelAssess2015                   |                                                              |
| @hamiltonRepresentationLearningGraphs2018          | Generate Time dependent Heterogeneous graphs using modularity optimization and multi-dependency sampling |
| SYN - @ghalebiDynamicNetworkModel2019              |                                                              |
| SBM - @lancichinettiBenchmarksTestingCommunity2009 | extracted from the dynamic Stochastic Block Model            |



### Real World

| Dataset                                                      | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Enron](https://www.cs.cmu.edu/~./enron/)                    | Includes: Persons, Email Categories, Sentiment, Email Content |
| [KIT](https://i11www.iti.kit.edu/en/projects/spp1307/emaildata) (dead) |                                                              |
| [Weibo](http://www.wise2012.cs.ucy.ac.cy/challenge.html)     | Includes: Persons, Tweets, Followers; **Excludes: Tweet Content** |
| [Digg](https://www.isi.edu/~lerman/downloads/digg2009.html)  | Includes: Persons, Stores, Followers, Votes; **Excludes: Content** |
| [Slashdot](http://snap.stanford.edu/data/soc-sign-Slashdot090221.html) | Includes: Persons, Votes; **Excludes: Content**              |
| [IMDB](https://paperswithcode.com/dataset/imdb-binary)       | Actor movie network; Content is implicitly defined           |
| [WIKI-RFA](https://snap.stanford.edu/data/wiki-RfA.html)     | Wikipedia Adminitrator Election; Network of Voters and Votees. Links are votes and vote comments |
| [FB-wosn](http://socialnetworks.mpi-sws.org/data-wosn2009.html) | User friendship links and User posts on users walls; **Excludes: Content** |
| [TweetUM](https://wis.st.ewi.tudelft.nl/research/tweetum/) (dead) | Twitter Tweets, User Profiles and Followers; Includes: Content |
| [Reddit Pushift](https://arxiv.org/abs/2001.08435)           | User Submissions and Posts on Subreddits; With timestamps    |
| [Bitcoin Trust Network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | Network Nodes and peer Ratings; With timestamps              |
| [LastFM1k](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) | User - Song Listen histories; With timestamps                |
| [MovieLens25M](https://grouplens.org/datasets/movielens/25m/) | Users and Movie Ratings; With timestamps                     |
| [Memetracker](https://snap.stanford.edu/data/memetracker9.html) |                                                              |
| [Rumor Detection](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168344) | Rumor Detection over Varying Time Windows; Twitter data; With timestamps |

## Dynamic Community Detection Methods

https://github.com/FanzhenLiu/Awesome-Deep-Community-Detection

### Link-based Methods

#### Community Detection

##### Modularity

* Measures the density of links inside communities compared to links between communities
* Range [-1/2, 1]

##### Louvain Method

The Louvain method is a popular algorithm to detect communities in large networks. It is a hierarchical clustering algorithm, as it recursively merges communities into single community and further executes modularity clustering on this condensed network.

@blondelFastUnfoldingCommunities2008

* Hierarchical algorithm
  * Starts with each node assigned to it’s own community
  * First small communities are found
    * For each node $i$ change in modularity is calculated for removing $i$ from its own community
    * And adding it to a neighbor community
    * Modularity change can be calculated incrementally (local)
  * Then produces condensed graph by merging communities (their nodes) into a single node
  * Repeats this process
* Optimizes for modularity
  * Is a heuristic algorithm
    * Since going through all possible assignments maximizing modularity is impractical
* TODO: Should I go deeper into calculation?

##### Label Propagation algorithm

* Algorithm to find communities in graph (very fast)
  * Uses only network structure as guide
  * Doesn’t require any priors (metrics)
  * Intuition:
    * Single label quickly becomes dominant in a group of densely connected nodes
    * But these labels have trouble crossing sparsely connected regions 
    * Nodes that end up with same label can be considered part of  same community
  * Algorithm:
    * Initialize each node to their own label
    * Propagate the labels, per iteration:
      * Each node updates its label to one that majority of its neighbors belong
      * Ties are broken deterministically
    * Stops when convergence is reached, or max iter
  * Preliminary solution can be assigned before run



#### Independent Community Detection and Matching

* (Instant Optimal Communities Discovery)
  * Works in two stages:
    * CD methods are applied directly to each snapshot  (identify stage)
    * Then the communities are matched between the snapshots (match stage)
  * Advantages:
    * Use of unmodified CD algorithms (built on top of exisiting work)
    * Highly parallelizable
  * Disadvantage:
    * Instability of community detection algorithms (may give **very** different results if network changes)
    * Difficult to distinguish between instability of algorithm and evolution of the network
* @wangCommunityEvolutionSocial2008 (core nodes / leader nodes)
  * Circumvents instability issue by studying most stable part of communities (community core nodes)
  * Datasets:
    * enron
    * imdb
    * caida
    * apache
  * Observations:
    * The social network scale inflates when it evolves
    * Members change dramatically and only a small portion exists stably
      * Therefore only a few can be relied on
  * Introduce algorithm CommTracker
    * Relies heavily on core nodes
    * Example: co-authorship community where core nodes represent famous professors
    * Core Node Detection Algorithm
      * Each node evaluates centrality of the nodes linked to it
      * If a node’s weight is higher than it’s neighbors - then its centrality is increased and neighbors decreased
        * The change value is set as difference is weight
      * Nodes with non-negative centrality are core nodes
    * Core-based Algorithm to track communities
      * Define a set of rules based on presence of core nodes 
      * To detect the evolu
* @greeneTrackingEvolutionCommunities2010 (similarity metric based - quality function)
  * Use quality funciton to quantify similarity between communities
  * Jaccard Similarity
  * MOSES: for CD
* @sunMatrixBasedCommunity2015 ()
  * Louvain algorithm for CD
  * Use correlation matrix for community matching
    * relation between t and t+1
  * Defined rules to detect evolution events based on matrix
* @rossettiANGELEfficientEffective2020 

#### Dependent Community Detection

* Dependent Community Detection / Temporal Trade-Off Communities Discovery
  * Use snapshots to detect communtities
  * To detect communities in current snapshot, rely on communities from the previous
  * Advantage:
    * Introduces temporal smoothness (fixes the instability problem mostly)
    * Does not diverge from the usual CD definition (search at each timestep)
  * Disadvantage:
    * Not parallelizable
    * Impacts long term coherence of dynamic communtities
    * Each steps experience substantial drift compared to what a static algorithm would find
* @heFastAlgorithmCommunity2015 (modified louvain)
  * Modify louvain algorithm to use previous communities
* @sunGraphScopeParameterfreeMining2007 (mdl based)
  * FIrst step: apply MDL method to encode graph with min number of bits
  * Divide network into multiple sequential segments
    * Jumps between segments mark evolution over time
  * Incrementally initialize using previous output
* @gaoEvolutionaryCommunityDiscovery2016 (leader nodes)
  * Propose evolutionary CD algorithm based on leader nodes
  * Each community is a set of followers around a leader
  * Present an updating strategy with temp info to get initial leader nodes
  * Leader nodes ensure temporal smoothneess
* @yinMultiobjectiveEvolutionaryClustering2021
  * Is a genetic algorithm
  * Look at the problem from an Evolutionary Clustering Perspective
    * Evolutionary as in Evolutionary Algorithms
    * Goal: detect community structure at the current time under guidance of one obtained immediately in the past
      * Fit observed data well
      * And keep historical consistency
  * Solve major drawbacks:
    * Absence of error correction - which may lead to result-drifting and error accumulation
    * NP-hardness of modularity based community detection
    * If initial community structure is not accurate, or the following - this may lead to the “result drift” and “error accumulation”
  * Propose DYN-MODPSO
    * Combine traditional evolutionary clustering with particle swarm algorithm
    * Propose a strategy to detect initial clustering - with insight of future ones
    * A new way to generate diverse population of individuals
  * Use previous work DYN-MOGA
    * Which introduces Multi-Objective optimization 
      * (Trade-off between - Both objectives are competitive)
      * Accuracy in current time step (CS - Community Score)
      * Similarity of two communities in consecutive steps (NMI - Similarity of two partitions)
  * Algorithm:
    * Initialization Phase:
      * De redundant random walk to initialize the initial population (with diverse individuals)
      * Random walk: 
        * Used to approximate probability of two nodes being linked
        * Then probability is sorted and is split on (based on Q metric optimizing modularity)
      * Represent as binary string encoding
    * Search Phase:
      * Particle swarm optimization - preserving global historically best positions
        * Already given initial swarms / clusterings
        * Identifies best positions for different nodes - and uses swarm velocity to interpolate between them
      * Builds good baseline clusterings
    * Crossover & Mutation Phase:
      * Uses custom operators MICO and NBM+ to improve global convergence
        * Cross-over operators - combining multiple individuals (parents)
      * Applies specific operators to maximize NMI of CS
  * Results:
    * Seems to perform well and be fast?

#### Simultaneous community detection

* Simultaneous community detection / Cross-Time Community Discovery
  * Doest consider independently the different steps of the network evolution
  * Advantage:
    * Doenst suffer from instability or community drift
    * More potential to deal with slow evolution and local anomalies
  * Disadvantage:
    * Not based on usual principle of a unique partition with each timestep
    * Cant handle real time community detection
  * Approaches fall in categories:
    * Fixed Memberships, Fixed Properties
      * Communities can change mambers and nodes cant disappear
      * @aynaudMultiStepCommunityDetection2011
    * Fixed Membership, Dynamic Properties
      * Memberships of communties cant change
      * But they are assigned a profile based on their temporal …
    * Evolving Membership, Fixed Properties
      * Allow membership change along time
      * Mostly SBM approaches
        * properties can not change over time
      * @ghasemianDetectabilityThresholdsOptimal2016
    * Evolving Membership, Evolving Properties
      * Dont impose any contraints
      * Edges are added between existing nodes between consequent snapshots
      * @muchaCommunityStructureTimeDependent2009
        * Modify louvain

#### Dynamic Community Detection on Temporal Networks (Evolution)

* Dynamic Community Detection on Temporal Networks / Online Approach
  * Works directly on temporal graphs
  * Network consists of series of changes
  * Algorithms are iterative
    * Find communties once
    * Then update them
* @zakrzewskaTrackingLocalCommunities2016 (leader nodes - rule based)
  * Dynamic set expandion
  * Indrementally update community
    * 1st step: initial community finding
    * iteration: new nodes are added
    * Fitness score is used to assign nodes to communtities
* @xieLabelRankStabilizedLabel2013 (label propagation)
  * LabelRankT - based on LabelRank (label propagation) algorithm
  * When network is updated, label of changed nodes are re-initialized
* @guoDynamicCommunityDetection2016 (based on distance dynamics)
  * Based on distance dynamics
  * Increments can be treated as network disturbance
  * Uses candidate set to pick pick ROI
    * Changes are made in communties
    * Distances are updated

#### Tacking Instability

* Temporal Smoothing:
  * Smoothing by bootstrap
  * Explicit smoothing: In definition
  * Implicit:
    * Reuse previous communities as seeds
    * Locally update communities
  * Global smoothing: search for communities by examining all steps of evolution

### Deep Methods

* * 

* Add additional links to the graph
  * Yoonsuk Kang
  
* Change distances within the graph

* Community Detection
  * @kangCommunityReinforcementEffective2021
    * Present a **Community Reinforcement** approach

      * Is CD algorithm agnostic

        * Shown in experiments - therefore the graph itself benefits
      * Reinforces the Graph by
      
        * Creating inter-community edges
        * Deleting intra-community edges
        * Determines the appropriate amount of reinforcement a graph needs
      * Which helps the dense inter-community and sparse intra-community property of the graph
        * Can effectively turn difficult-to-detect community situation into that of easy-to-detect communities
      * Challenges:
        * Needs to be  unsupervised (doesn't need community annotations to work)
        * Appropriate amount of reinforcement needs to be determined (otherwise noise is introduced)
        * Needs to be fast, checking every possible edge is infeasible
      * Methodology:
        * Edge Addition / Deletion
          * Based on node similarity of connected pairs
            * Similar nodes are likely to be in a community (intra edges)
            * Dissimilar ones are less likely to be in a community (inter edges)
          * Employ graph embedding to generate topological embeddings
            * Adamic/Adar (local link-based)
            * SimRank (gloabl link-based)
            * Node2vec graph embedding based
          * Predict similarities and bucket them
            * Use similarity buckets to select intra and inter edges - and to tune the model
            * Buckets are selected on how well they predict current edges
        * Detect the right amount of addition
          * Use a gradual reinforcement strategy by generating a series of graphs (adding top x inter and removing intra edges)
          * Pick the best graph using a scoring function (modularity)
          * Simply run CD over the graph and see
        * Reducing comutational overhead
          * Using a greedy similarity computation
          * Prefer nodes which are likely to be in same community of inter similarity detection
      * Tests results on:
        * Synthetic Graphs: LFR
        * Real world graphs: Cora, Siteseer, DBLP, Email
  * @huangInformationFusionOriented2022
    * Their own made dataset: https://github.com/MingqingHuang-SHU/HRTCD
    * Not really graph augmenting?
    * See (Rumor Detection) dataset also
  * @luberCommunityDetectionHashtagGraphsSemiSupervised2021
    * short
  * @jiaCommunityGANCommunityDetection2019
    * has code
  * @rozemberczkiGEMSECGraphEmbedding2019
    * hh
  
* Dynamic Community Detection

  * @wangVehicleTrajectoryClustering2020
    * Transform task of trajectory clustering into one of Dynamic Community Detection
      * Discretize the trajectories by recording entity their current neigbors at each time interval 
      * Edge streaming network is created
    * Use representation learning to learn node their embeddings
      * Use dyn walks to perform random walks in time dimenstion
      * Use negative sampling to avoid the softmax cost
    * Then use K-means to find the communities
      * Try K-means, K-medioids and GMM (Gaussian Mixture Models)
      * Initalize the centers at the previous timestamp centers
    * Use quality measures to establish quality of results

  * @maCommunityawareDynamicNetwork2020 (use as baseline?)
    * Define communities in terms of large and small scale communities
    * They propose a method for dynamic *community aware* network representation learning
      * By creating a unified objective optimizing stability of communities, temporal stability and structure representation
      * Uses both first-order as well as second order proximity for node representation learning
    * They define community representations as average of their members
      * Adopt a stacked autoencoder to learn low-dimensional (generic) representations of nodes
    * They define loss in terms of:
      * Reconstruction Error: How well the graph can be reconstructed from the representation
      * Local structure preservation: Preservation of homophiliy - connected nodes are similar
      * Community evolution preservation: Preservation of smoothness of communities in time at multiple granularity levels
    * The communities are initialized using classical methods:
      * First large communities are detected using Genlouvin (fast and doesnt require priors)
      * Then small scale communities are detected using k-means by defining a max community size $w$
        * Which provides more fine tuned communities
    * Using the initial embeddings the temporal embeddings are optimized 
      * Done by optimizing all at once - therefore maintaining the stability
      * And use of the mentioned combined objective
    * Though they present / evaluate their algorithm in terms of Dynamic Representation Algorithms
      *  Therefore the actual quality of communities remains to be known
  * @wangEvolutionaryAutoencoderDynamic2020
    * Approach is similar to to @maCommunityawareDynamicNetwork2020
    * Defines a unified objective where 
      * community characteristics
      * previous clustering 
      * are incorporated as a regularization term
    * They argue that real world networks are non-linear in nature and classical approaches can not capture this
      * Autoencoders can though
    * Methodology:
      * Construct a similarity matrix using Dice Coefficient (handles varying degrees well)
      * Apply stacked (deep) autoencoders to learn the low-dimensional representation
      * Characterizes tradeoff between two costs:
        * Snapshot cost (SC): 
          * Reconstruction loss
          * Node embedding similarity (homophiliy) between connected nodes
          * Node embedding similarity (homophiliy) between nodes in same community
        * Temporal cost (TC)
          * Temporal smoothness of node embeddings
      * Adopt K-means to discover community structures
  * @faniUserCommunityDetection2020
    * Propose a new method of identifying user communities through multimodal feature learning:
      * learn user embeddings based on their **temporal content similarity**
        * Base on topics of interest
        * Users are considered like-minded if they are interested in similar topics at similar times
        * Learn embeddings using a context modelling approach
      * learn user embeddings based on their **social network connections**
        * Use GNN which works as a skip-gram like approach by generating context using random walks
      * **interpolate** temporal content-based embeddings and social link-based embeddings
    * Then they use these multimodal embeddings to detect dynamic communities\
      * Communities are detected on a modified graph 
        * Weights are set given embedding similarity
        * Communities are detected using louvain methods
      * Then test their approach on specific tasks such as
      * News recommendation
      * User for content prediction
    * Note: **This approach detects static communities**
      * But the communities implicitly take time into account

* General Strategy:
  * Represent
  * Recluster
    * Construct a new graph using deep based distances, and use Link Based CD
    * Cluster with a clustering algorithm
  * 

## Graph Representation Learning



* components
* problems with current solutions
* datasets [@rossettiCommunityDiscoveryDynamic2018]
* DCD is seen as the hardest problem within Social Network Analysis. Reason for this is mostly because DCD, unlike CD, also involves tracking the found communities over time which brings 

# Approach





# Planning

abc

