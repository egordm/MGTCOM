## Evaluation

% * Community definition varies a lot
% * Describe how detection and result tracking are evaluation
% * Detection and Tracking are evaluated separately

As described in the previous sections, the definition for both community and dynamic community may be quite ambiguous. In this section we will cover how detection and tracking results can be evaluated in a lesser ambiguous setting to compare various approaches. To disambiguate the process, during evaluation, the resemblance/detection and matching/tracking tasks are considered as separate evaluation tasks.



### Annotated

% * Ground Truth communities - compare againts them
% * Use NMI or other set overlap method
% * Talk about NF1 @rossettiNovelApproachEvaluate2016
% * Usually no planted communities:
%   * Synthetic datasets
%   * Use annotated features

Evaluation of detected (dynamic) communities becomes much easier when the *ground truth communities* are provided. The evaluation is then done by comparing the difference between the produced communities and the effective ones. To perform this comparison, information theory based metric Normalized Mutual Information (NMI) is used which converts community sets to bit-strings and quantifies the “amount of information” can be obtained about one community by observing the other [@lancichinettiDetectingOverlappingHierarchical2009].

A possible drawback of this measure is that its complexity is quadratic in terms of identified communities. In [@rossettiNovelApproachEvaluate2016] alternative measure (NF1) with linear complexity is introduced which similarly to F1 score uses the trade-off between precision and recall (of the average of harmonic means) of the matched communities. In the follow-up work [@rossettiANGELEfficientEffective2020] the authors describe a way to apply this measure within the context of DCD by calculating this score for all the snapshots and aggregating the results into one single measure.

In real-world there are usually no ground truth communities. Therefore this approach is usually applied on synthetic datasets where the communities and their dynamicity is sampled from a distribution. Alternative approach some papers take is by defining ground truth communities using the metadata and node attributes present within the datasets. Some datasets may include annotated communities, but this is not common within DCD datasets.



### Metric based

% * When Ground Truth Communities don't exist
% * Network-based measures

Another way to evaluate and compare different CD algorithms without knowing ground truth communities is using a quality function. Modularity is the most widely used measure [@newmanFastAlgorithmDetecting2004], since it measures the strength of division of a network into modules. Networks with high modularity have dense connections between the nodes within the modules, and sparse connections between nodes in different modules. Other measures are used as well including:

* Conductance: the percentage of edges that cross the cluster border
* Expansion: the number of edges that cross the community border
* Internal Density: the ratio of edges within the cluster with respect to all possible edges
* Cut Ratio and Normalized Cut: the fraction of all possible edges leaving the cluster
* Maximum/Average ODF: the maximum/average fraction of nodes’ edges crossing the cluster border



% Representation-/Proximity- based measures



% @wangVehicleTrajectoryClustering2020
% 
% * Use taxi dataset with license plates
% * Compare to other deep GNN - they only learn static representation
% * Metrics:
%   * **Silhouette Coefficient** (SC) - range [-1, 1]
%     * $S(i)=\frac{b(i)-a(i)}{\max \{a(i), b(i)\}}$
%     * $a$ avg distance between node and neighbors in cluster
%     * $b$ is min val of avg distances between node and other clusters
%   * **Davies-Bouldin Index** (DBI)
%     * $D B I=\frac{1}{N} \sum_{i=1}^{N} \max_{j \neq i}\left(\frac{\overline{S_{i}}+\overline{S_{j}}}{\left\|w_{i}-w_{j}\right\|_{2}}\right)$
%     * $\bar{S_i}$: avg distance of nodes in cluster $i$ to centroid of cluster $i$
%     * $w_i$ is the centroid of cluster $w_i$
%     * It is the ratio of the sum of the average distance to the distance between the centers of mass of the two clusters
%     * The closer the clustering result is with the inner cluster, and the farther the different clusters, the better the result
%   * **Calinski-Harabaz Index** (CHI): Ratio of the between-cluster variance and within-cluster variance
%     * $C H I=\frac{\operatorname{tr}\left(B_{k}\right)}{\operatorname{tr}\left(W_{k}\right)} \frac{m-k}{k-1}$
%     * $m$ number of nodes, $k$ number of clusters,
%     * $B_k$ is covariance matrix between the clusters
%     * $W_k$ is covariance matrix between the data in the cluster
%     * $tr$ is trace of the matrix



% @mrabahRethinkingGraphAutoEncoder2021
% 
% * Accuracy:
% * NMI
% * ARI:
% 
% @huangInformationFusionOriented2022
% 
% * Based on link prediction or friend recommendation
% * Precision
% * Recall
% * F-score
% * normalized discounted cumulative gain (nDCG)
% * mean reciprocal rank (MRR)



### Task specific

% * @peelGroundTruthMetadata2017 Criticize methods that define their own community criteria
%   * No free lunch - solving task for your definition wont solve all CD problems
%   * When CD algorithm, it is indistinguishable from possibilities: irrelevant metadata, orthogonal data, network lacks structure.
%   * Evaluate on tasks and usecases and not based on a single measure

In [@peelGroundTruthMetadata2017] the authors criticize **these** evaluation approaches by proving that they introduce severe theoretical and practical problems. For one, they prove the no free lunch theorem for CD, ie. they prove that algorithmic biases that improve performance on one class of networks must reduce performance on others. Therefore, there can be no algorithm that is optimal for all possible community detection tasks, as quality of communities may differ by the optimized metrics. Additionally, they demonstrate that when a CD algorithm fails, the poor performance is indistinguishable from any of the three alternative possibilities: (i) the metadata is irrelevant to the network structure, (ii) the metadata and communities capture different aspects of network structure, (iii) the network itself lacks structure. Therefore, which community is optimal should depend on it’s subsequent use cases and not a single measure.



% @faniUserCommunityDetection2020
% 
% * Compare against Static Content Based CD Algorithms
% * Content Based Community Detection
% * Compare against Static Link Based CD Algorithms
% * Compare against Multimodal Based CD Algorithms
% * Problems:
%   * Absence of ground truth communities
%   * Modularity cant be used - based on explicit links betwene users (structural)
%     * Doesnt account for content at all
% * Solutions: Application level evaluation
%   * A user community detection method is considered to have better quality iff its output communities improve an underlying application
%   * **News recommendation** (in time)
%     * Curate dataset of news articles mentioned by users (user mention means user interest)
%     * Methodology:
%       * Detect communities and assign them a topic of interest at a time
%       * Topic is average of user interests
%       * All news articles are ranked based on their similarity with the overall topic (in time)
%       * Each member in community is recommended the ranked list
%     * Metrics: (stadard info retreval metrics)
%       * Precision at rank $k$ ($P_k$)
%         * $\mathrm{P}_{k}=\frac{1}{|\mathrm{U}|} \sum_{u \in \mathbb{U}} \frac{t p_{u}}{k}$
%         * $u$ is user
%       * Mean Reciprocal Rank (MRR)
%         * $\mathrm{MRR}=\frac{1}{|\mathbb{U}|} \sum_{u \in \mathbb{U}} \frac{1}{\operatorname{rank}_{u}}$
%         * First position correct result occurs in list
%       * Success at rank $k$ ($S_k$)
%         * Probability that at least one correct item is within a top-k list
%         * $\mathrm{S}_{k}=\frac{1}{|\mathbb{U}|} \sum_{u \in \mathcal{U}}\left(\operatorname{rank}_{u} \leq k\right)$
%   * **User Prediction**
%     * Goal: Predict which users posted a news article $a$ at time $t$
%     * Methodology:
%       * Find closest community to the article in terms of interest at time $t$ (cosine sim)
%       * Members of community are predicted users
%     * Same reasoning as news prediction
%     * Metrics (classificiation metrics)
%       * Precision, Recall, F-measure



% @maCommunityawareDynamicNetwork2020
% 
% * Use both synthetic and real world datasets
% * Use not perse community detection baselines
% * Define auxilary helper tasks in context of *community aware* **Deep Network Embedding**:
%   * **Network Reconstruction**: Evaluates model on ability of reconstructing link structures of the network
%     * Average reconstruction precision is measured
%     * This is done for each timestamp
%     * For each node, the nearest embedding neighbors are used as predicted links
%   * **Link Prediction**:
%     * Prediction of existence of links between nodes in the next timestamps
%     * Based on representation in the current timestamp
%   * **Network Stabilization**: evaluates preformance of DNE on stabilization of embedding
%     * dynamic network should have similar evolutionary patterns in both the learned low-dimensional representation and the
%       network representation over time
%     * evaluates the evolution ratio of the low-dimensional node representations to the network representations at $a$ th timestamp
%     * $p_{s}^{a}=\frac{\left(\left\|\mathrm{H}^{a+1}-\mathrm{H}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}^{a+1}-\mathrm{A}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}^{a}\right\|_{2}^{2}} .$
%   * **Community Stabilization**: evaluates stability of communities in dynamic networks on the embedded low-dimensional representations
%     * Evaluates communty evolution ratio to network representation evolution between subsequent timestamps
%     * Lower values point to more stable communities and are better
%     * $p_{c}^{a}=\sum_{k=1}^{q}\left(\frac{\left(\left\|\mathrm{H}_{c_{k}}^{a+1}-\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}_{c_{k}}^{a+1}-\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}}\right) / q$
% * Network is first fine-tuned on each of the tasks
% * Note: the evalution is at graph level since their methods are spectral GAE based





## Datasets

### Synthetic Datasets

| Paper                                              | Description                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| @lancichinettiBenchmarkGraphsTesting2008           | Static networks (widely used)                                                                            |
| @greeneTrackingEvolutionCommunities2010            | Generate Graphs based on Modularity measure                                                              |
| @granellBenchmarkModelAssess2015                   |                                                                                                          |
| @hamiltonRepresentationLearningGraphs2018          | Generate Time dependent Heterogeneous graphs using modularity optimization and multi-dependency sampling |
| SYN - @ghalebiDynamicNetworkModel2019              |                                                                                                          |
| SBM - @lancichinettiBenchmarksTestingCommunity2009 | extracted from the dynamic Stochastic Block Model                                                        |

### Real World Datasets

| Dataset                                                                                      | Description                                                                                      |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [Enron](https://www.cs.cmu.edu/~./enron/)                                                    | Includes: Persons, Email Categories, Sentiment, Email Content                                    |
| [KIT](https://i11www.iti.kit.edu/en/projects/spp1307/emaildata) (dead)                       |                                                                                                  |
| [Weibo](http://www.wise2012.cs.ucy.ac.cy/challenge.html)                                     | Includes: Persons, Tweets, Followers; **Excludes: Tweet Content**                                |
| [Digg](https://www.isi.edu/~lerman/downloads/digg2009.html)                                  | Includes: Persons, Stores, Followers, Votes; **Excludes: Content**                               |
| [Slashdot](http://snap.stanford.edu/data/soc-sign-Slashdot090221.html)                       | Includes: Persons, Votes; **Excludes: Content**                                                  |
| [IMDB](https://paperswithcode.com/dataset/imdb-binary)                                       | Actor movie network; Content is implicitly defined                                               |
| [WIKI-RFA](https://snap.stanford.edu/data/wiki-RfA.html)                                     | Wikipedia Adminitrator Election; Network of Voters and Votees. Links are votes and vote comments |
| [FB-wosn](http://socialnetworks.mpi-sws.org/data-wosn2009.html)                              | User friendship links and User posts on users walls; **Excludes: Content**                       |
| [TweetUM](https://wis.st.ewi.tudelft.nl/research/tweetum/) (dead)                            | Twitter Tweets, User Profiles and Followers; Includes: Content                                   |
| [Reddit Pushift](https://arxiv.org/abs/2001.08435)                                           | User Submissions and Posts on Subreddits; With timestamps                                        |
| [Bitcoin Trust Network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)            | Network Nodes and peer Ratings; With timestamps                                                  |
| [LastFM1k](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)                      | User - Song Listen histories; With timestamps                                                    |
| [MovieLens25M](https://grouplens.org/datasets/movielens/25m/)                                | Users and Movie Ratings; With timestamps                                                         |
| [Memetracker](https://snap.stanford.edu/data/memetracker9.html)                              |                                                                                                  |
| [Rumor Detection](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168344) | Rumor Detection over Varying Time Windows; Twitter data; With timestamps                         |
