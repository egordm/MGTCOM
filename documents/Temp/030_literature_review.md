# Literature review

The problem of dynamic community detection was noticed quite early on in within the SNA community and a considerable amount of research have been made in order to provide a comprehensive analysis of the network. While the said research was mostly focused on discovery of communities using topologically-based features and node connectivity, the covered methods did research the limitations and challenges posed by a temporal context. 

In the recent years, significant developments have been made in the space of deep learning. Mainly in the development of new deep learning methods capable of learning graph-structured data  [@bronsteinGeometricDeepLearning2017; @hamiltonRepresentationLearningGraphs2018; @kipfSemiSupervisedClassificationGraph2017] which is fundamental for SNA. Various of problems within the field have been revisited, including the community detection problems. The approaches have been expanded by incorporation of more complex features, solving the problems concerning multi-modality and introduction of unsupervised learning. 

Despite this resurgence, the DCD problem has received little attention. Though a few efforts have been made to incorporate the deep learning methods by introducing content-based similarity dynamic and usage of graph representation based CD algorithms within a temporal context, the current state of the art leaves a lot to be desired. 

We structure this literature review as follows: first we describe the various interpretations of the community structure @community-structures, explore the current datasets and evaluation methods used for benchmarking of the current DCD methods @evaluation-methods. Then, we dive in the current state of the art works on DCD by discussing both “classic” methods and novel deep learning based methods +@datasets. Finally, we discuss the current advances within the graph representation learning [] and community detection [] approaches.

## Community Structures

Communities in real-world networks can be of different kinds: disjoint (think of students belonging to different educational institutions), overlapping (person having membership in different social groups) and hierarchical (components of a car). One of the main reasons behind the complexity of CD is the unclear definition what a community actually is.

The *link-based* (referred to as classic) community detection methods intuitively describe communities as groups of nodes within a graph, such that the intra-group connections are denser than the intergroup ones. This definition is primarily based on the *homophily* principle, which refers to the assumption that similar individuals are those that are densely connected together. Therefore, these kind of methods look for sub-graph structures such as cliques and components that identify connectedness within the graph structure to represent the communities. 

Unfortunately, in most cases link-based methods fall short to identity communities of similar individuals. This is mainly due to two facts: (i) many similar individuals in a social network are not explicitly connected together, (ii) an explicit connection does not necessarily indicate similarity, but may explained by sociological processes such as conformity, friendship or kinship [@diehlRelationshipIdentificationSocial2007; @faniUserCommunityDetection2020].

Because of this, and with success of (deep learning) representation based methods, the more recent works define communities as a distribution over $d$-dimensional space which may span over both topological as well as content-based features [@cavallariLearningCommunityEmbedding2017]. Similarly, also hybrid methods are introduced which combine link-based approach with content representation learning methods [@faniUserCommunityDetection2020].

A more general definition is introduced in [@cosciaClassificationCommunityDiscovery2011] to create an underlying concept generalizing all variants found in the literature. In link-based methods, a direct connection is considered as a particular and very important kind of action, while newer methods also consider content or interest overlap.

> *Definition (Community). A community in a complex network is a set of entities that share some closely correlated sets of actions with the other entities of the community.* 

* TODO: Content Based Methods: (similarity idea)
  * Modeled communities **based on topics of interest** through a community-user-topic generative process
  * **Communities** are **formed around multiple correlated topics** where each topic can be reused in several different communities
  * User can be a member of different communities but with varying degrees of membership
  * **Distributional semantics** states that words that occur in similar contexts are semantically similar
  * **Recommendation Point of view**

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

## Evaluation methods

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
  * 



## Datasets

### Synthetic

| Paper                                     | Description                                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| @greeneTrackingEvolutionCommunities2010   | Generate Graphs based on Modularity measure                  |
| @granellBenchmarkModelAssess2015          |                                                              |
| @hamiltonRepresentationLearningGraphs2018 | Generate Time dependent Heterogeneous graphs using modularity optimization and multi-dependency sampling |



### Real World

| Dataset                                                      | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Enron](https://www.cs.cmu.edu/~./enron/)                    | Includes: Persons, Email Categories, Sentiment, Email Content |
| [KIT](https://i11www.iti.kit.edu/en/projects/spp1307/emaildata) (dead) |                                                              |
| [Weibo](http://www.wise2012.cs.ucy.ac.cy/challenge.html)     | Includes: Persons, Tweets, Followers; **Excludes: Tweet Content** |
| [Digg](https://www.isi.edu/~lerman/downloads/digg2009.html)  | Includes: Persons, Stores, Followers, Votes; **Excludes: Content** |
| [Slashdot](http://snap.stanford.edu/data/soc-sign-Slashdot090221.html) | Includes: Persons, Votes; **Excludes: Content**              |
| [IMDB](https://paperswithcode.com/dataset/imdb-binary)       | Actor movie network; Content is implicitly defined           |
| [WIKI-RFA](https://snap.stanford.edu/data/wiki-RfA.html)     | Network of Voters and Votees. Links are votes and vote comments |
| [FB-wosn](http://socialnetworks.mpi-sws.org/data-wosn2009.html) | User friendship links and User posts on users walls; **Excludes: Content** |
| [TweetUM](https://wis.st.ewi.tudelft.nl/research/tweetum/) (dead) | Twitter Tweets, User Profiles and Followers; Includes: Content |
| [Reddit Pushift](https://arxiv.org/abs/2001.08435)           | User Submissions and Posts on Subreddits; With timestamps    |
| [Bitcoin Trust Network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | Network Nodes and peer Ratings; With timestamps              |
| [LastFM1k](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) | User - Song Listen histories; With timestamps                |
| [MovieLens25M](https://grouplens.org/datasets/movielens/25m/) | Users and Movie Ratings; With timestamps                     |



## Dynamic Community Detection Methods



### Classical Methods



### Deep Methods



## Graph Representation Learning



* components
* problems with current solutions
* datasets [@rossettiCommunityDiscoveryDynamic2018]
* DCD is seen as the hardest problem within Social Network Analysis. Reason for this is mostly because DCD, unlike CD, also involves tracking the found communities over time which brings 

* 