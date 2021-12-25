## Evaluation

% * Community definition varies a lot
% * Describe how detection and result tracking are evaluation
% * Detection and Tracking are evaluated separately

In the previous section we have described the different variations of community structure defintions as well as the approaches used for detecting these communities. In this sections we will cover how the found dynamic community structures can be evaluated in a more general setting to allow comparison of different approaches. While dynamic community detection problem can be seen as two tasks (resemblance/detection and matching/tracking) these two should separate evaluation tasks. 

In the following sections we cover both of the evaluation tasks by classifying approaches used in the literature in three classes. Namely, annotated, metric-based and task specific.



### Annotated

% * Ground Truth communities - compare against them
% * Use NMI or other set overlap method
% * Talk about NF1 @rossettiNovelApproachEvaluate2016
% * Usually no planted communities:
%   * Synthetic datasets
%   * Use annotated features

Evaluation of detected (dynamic) communities becomes much easier when the *ground truth communities* are provided. The evaluation is then done by comparing the difference between the produced communities and the effective ones. To perform this comparison, information theory based metric Normalized Mutual Information (NMI) is used which converts community sets to bit-strings and quantifies the “amount of information” can be obtained about one community by observing the other [@lancichinettiDetectingOverlappingHierarchical2009].

A possible drawback of this measure is that its complexity is quadratic in terms of identified communities. In [@rossettiNovelApproachEvaluate2016] alternative measure (NF1) with linear complexity is introduced which similarly to F1 score uses the trade-off between precision and recall (of the average of harmonic means) of the matched communities. In the follow-up work [@rossettiANGELEfficientEffective2020] the authors describe a way to apply this measure within the context of DCD by calculating this score for all the snapshots and aggregating the results into one single measure.

% * Accuracy: @mrabahRethinkingGraphAutoEncoder2021
% * NMI: @mrabahRethinkingGraphAutoEncoder2021, @jiaCommunityGANCommunityDetection2019, @yangCommunityAffiliationGraphModel2012
% * ARI: @mrabahRethinkingGraphAutoEncoder2021, @luoDetectingCommunitiesHeterogeneous2021
% * F1: @jiaCommunityGANCommunityDetection2019, @yangCommunityAffiliationGraphModel2012
% * Omega Index: @yangCommunityAffiliationGraphModel2012
%   * is the accuracy on estimating the number of communities that each pair of nodes shares

Aside from NMI other measures are employed such as Jaccard Coeficient, Accuracy and Rand-Index measuring community overlap [@yangGraphClusteringDynamic2017; @mrabahRethinkingGraphAutoEncoder2021; @luoDetectingCommunitiesHeterogeneous2021], Overlapping-NMI [@yeDeepAutoencoderlikeNonnegative2018] and Omega-Index measuring is the accuracy on estimating the number of communities that each pair of nodes shares [@yangCommunityAffiliationGraphModel2012], 

In real-world there are usually no ground truth communities. Therefore this approach is usually applied on synthetic datasets where the communities and their dynamicity is sampled from a distribution. Alternative approach some papers take is by defining ground truth communities using the metadata and node attributes present within the datasets. Some datasets may include annotated communities, but this is not common within DCD datasets.



### Metric based

% * When Ground Truth Communities don't exist
% * Network-based measures

Another way to evaluate and compare different CD algorithms without knowing ground truth communities is using a quality function. 

#### Network-based metrics

The first first group of measures we consider operate directly on the network structure. They are most commonly used to evaluate link-based methods as their results are network partitioning sets. Modularity is the most widely used measure [@newmanFastAlgorithmDetecting2004; @suComprehensiveSurveyCommunity2021], since it measures the strength of division of a network into modules. Networks with high modularity have dense connections between the nodes within the modules, and sparse connections between nodes in different modules. Other measures are used as well including:

* **Conductance**: the percentage of edges that cross the cluster border
* **Expansion**: the number of edges that cross the community border
* **Internal Density**: the ratio of edges within the cluster with respect to all possible edges
* **Cut Ratio and Normalized Cut**: the fraction of all possible edges leaving the cluster
* **Maximum/Average ODF**: the maximum/average fraction of nodes’ edges crossing the cluster border
* **Triangle Participation Ratio TPR**: measures fraction of triads within the community. A higher TPR indicates a denser community structure

% See: @suComprehensiveSurveyCommunity2021 as it has a well curated collection of evaluation metrics

#### Proximity-based measures

Proximity-based measures are often used to evaluate clustering tasks, but are also often used to evaluate representation-based CD methods since there is a large overlap in their methodology. Additionally, representation based approaches have a benefit of being able to quantify both entities and communities as a d-dimensional vector enabling a more direct comparison of the two [@wangVehicleTrajectoryClustering2020]. The most common measures include:

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

* **Silhouette Coefficient**: Is defined as $S(i)=\frac{b(i)-a(i)}{\max \{a(i), b(i)\}}$ where $a(i)$ defines the mean distance from node $i$  to other nodes in the same cluster while $b(i)$ is mean distance to any node not in the same cluster. It measures cohesion of a cluster/community and indicates how well a node is matched to its own cluster.
* **Davies-Bouldin Index**: Is the ratio of the sum of the average distance to the distance between the centers of mass of the two clusters. In other words, it is defined as a ratio of within cluster, to the between cluster separation. The index is defined as an average over all the found clusters, and is therefore also a good measure to deciding how many clusters should be used.
* **Calinski-Harabasz Index**: Is similarly the ratio of the between-cluster to the within-cluster variance. Therefore it measures both cohesion (how well its members fit the cluster) as well as compares it to other clusters (separation).
  
  

#### Stability-based measures

% * Stability-based measures evaluate temporal stability and consistency of the network
%   * The ones we consider were introduced in @maCommunityawareDynamicNetwork2020
%   * 
% * **Network Stabilization**: evaluates performance of DNE on stabilization of embedding
%   * dynamic network should have similar evolutionary patterns in both the learned low-dimensional representation and the
%     network representation over time
%   * evaluates the evolution ratio of the low-dimensional node representations to the network representations at $a$-th time stamp
%   * $p_{s}^{a}=\frac{\left(\left\|\mathrm{H}^{a+1}-\mathrm{H}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}^{a+1}-\mathrm{A}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}^{a}\right\|_{2}^{2}} .$
% * **Community Stabilization**: evaluates stability of communities in dynamic networks on the embedded low-dimensional representations
%   * Evaluates community evolution ratio to network representation evolution between subsequent time stamps
%   * Lower values point to more stable communities and are better
%   * $p_{c}^{a}=\sum_{k=1}^{q}\left(\frac{\left(\left\|\mathrm{H}_{c_{k}}^{a+1}-\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}_{c_{k}}^{a+1}-\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}}\right) / q$
% * Note: the evaluation is at graph level since their methods are spectral GAE based

To evaluate temporal stability and consistency of the node and community structures, measures based on temporal smoothness are proposed in @maCommunityawareDynamicNetwork2020. These measures compare the evolution rate of network structures against the evolution rate of the whole network given node embeddings between two consecutive snapshots. The intuition is that rapidly evolving structures relative to global evolution rate are temporally unstable and thus of low quality. Metrics proposed include:

* **Network Stability**: Is defined as +@eq:networkstability and evaluates the evolution ratio of the low-dimensional node representations to the network representations between snapshots at $a$-th time stamp.
* **Community Stability**: Is defined as +@eq:communitystability and computes stability of communities in dynamic networks on the embedded low-dimensional representations. It is represented as evolution ratio of the low-dimensional community representations to the network representations between snapshots at $a$-th time stamp.

$$
p_{s}^{a}=\frac{\left(\left\|\mathrm{H}^{a+1}-\mathrm{H}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}^{a+1}-\mathrm{A}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}^{a}\right\|_{2}^{2}}
$$ {#eq:networkstability}

$$
p_{c}^{a}=\sum_{k=1}^{q}\left(\frac{\left(\left\|\mathrm{H}_{c_{k}}^{a+1}-\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}}{\left(\left\|\mathrm{~A}_{c_{k}}^{a+1}-\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}\right) /\left\|\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}}\right) / q
$$ {#eq:communitystability}

In this case $\left\|\mathrm{H}_{c_{k}}^{a+1}-\mathrm{H}_{c_{k}}^{a}\right\|_{2}^{2}$ represents the normalized euclidean distance between the two clusters while $\left\|\mathrm{~A}_{c_{k}}^{a+1}-\mathrm{A}_{c_{k}}^{a}\right\|_{2}^{2}$ represents the normalized euclidean distance between the adjacency matrices of the two clusters. For network stability no such grouping is done and evaluation is performed on global representation and adjacency matrices. 

### Task specific

% * @peelGroundTruthMetadata2017 Criticize methods that define their own community criteria
%   * No free lunch - solving task for your definition wont solve all CD problems
%   * When CD algorithm, it is indistinguishable from possibilities: irrelevant metadata, orthogonal data, network lacks structure.
%   * Evaluate on tasks and use-cases and not based on a single measure

In [@peelGroundTruthMetadata2017] the authors criticize annotation and metric based CD evaluation approaches by proving that they introduce severe theoretical and practical problems. For one, they prove the no free lunch theorem for CD, ie. they prove that algorithmic biases that improve performance on one class of networks must reduce performance on others. Therefore, there can be no algorithm that is optimal for all possible community detection tasks, as quality of communities may differ by the optimized metrics. Additionally, they demonstrate that when a CD algorithm fails, the poor performance is indistinguishable from any of the three alternative possibilities: (i) the metadata is irrelevant to the network structure, (ii) the metadata and communities capture different aspects of network structure, (iii) the network itself lacks structure. Therefore, which community is optimal should depend on it’s subsequent use cases and not a single measure.

% * Problems:
%   * Absence of ground truth communities
%   * Modularity cant be used - based on explicit links between users (structural)
%     * Doesn't account for content at all
% * Solutions: Application level evaluation
%   * A user community detection method is considered to have better quality iff its output communities improve an underlying application

To address this issue, is common for to evaluate algorithm on both earlier described approaches as auxiliary tasks. The general sentiment behind it is, that communities have better quality if they improve an underlying application. In the following sections we describe a few commonly used auxiliary tasks in literature.



#### Link-prediction

% * Link-prediction is a common task to evaluate embedding quality within Graph Representation learning
%   * It usually given an existing graph links are split into a train and test set
%   * The model is trained on the test set, while the edges in train set are predicted and evaluated using classification metrics
%   * In context of dynamic community detection some changes can be applied 
% * **User Prediction** @faniUserCommunityDetection2020
%   * Goal: Predict which users posted a news article $a$ at time $t$
%   * Methodology:
%     * Find closest community to the article in terms of interest at time $t$ (cosine sim)
%     * Members of community are predicted users
%   * Same reasoning as news prediction
%   * Metrics (classification metrics)
%     * Precision, Recall, F-measure
% * **Link Prediction**: @maCommunityawareDynamicNetwork2020
%   * Prediction of existence of links between nodes in the next time stamps
%   * Based on representation in the current time stamp

A common way to evaluate quality of extracted node embeddings within Graph Representation learning is using the link-prediction task. As link-prediction can be done in unsupervised manner, it does not require additional labeling for evaluation. The edge set of the input network is split into a test set on which the model is trained, and a test set on which is used to compare the predicted links against. For node representation learning the predictions are defined by proximity between the nodes and a threshold. To quantify the quality of the results classification metrics are usually employed such as accuracy, precision, recall and f-score.

In context of CD and DCD the link-prediction is usually modified to measure the predicting capability of the community embeddings. @faniUserCommunityDetection2020 define a user prediction task to predict which users posted a certain news article at a certain time. Their methodology is, given a news at time $t$, find the closest community to the article in representational similarity space. All members of the given community are seen as predicted users over which the classification metrics are calculated. Similarly @maCommunityawareDynamicNetwork2020 modify the task by predicting whether the edges will still exist within the next time stamp to also quantify temporal prediction capability of the trained embeddings.



#### Recommendation Tasks

% * Generally identified by the fact that they do population based recommendation
% * **News recommendation** (in time) @faniUserCommunityDetection2020
%   * Curate dataset of news articles mentioned by users (user mention means user interest)
%   * Methodology:
%     * Detect communities and assign them a topic of interest at a time
%     * Topic is average of user interests
%     * All news articles are ranked based on their similarity with the overall topic (in time)
%     * Each member in community is recommended the ranked list
%   * Metrics: (stadard info retreval metrics)
%     * Precision at rank $k$ ($P_k$)
%       * $\mathrm{P}_{k}=\frac{1}{|\mathrm{U}|} \sum_{u \in \mathbb{U}} \frac{t p_{u}}{k}$
%       * $u$ is user
%     * Mean Reciprocal Rank (MRR)
%       * $\mathrm{MRR}=\frac{1}{|\mathbb{U}|} \sum_{u \in \mathbb{U}} \frac{1}{\operatorname{rank}_{u}}$
%       * First position correct result occurs in list
%     * Success at rank $k$ ($S_k$)
%       * Probability that at least one correct item is within a top-k list
%       * $\mathrm{S}_{k}=\frac{1}{|\mathbb{U}|} \sum_{u \in \mathcal{U}}\left(\operatorname{rank}_{u} \leq k\right)$
% * @rozemberczkiGEMSECGraphEmbedding2019
%   * Music recommendation
% * Based on link prediction or friend recommendation @huangInformationFusionOriented2022
%   * Precision
%   * Recall
%   * F-score
%   * normalized discounted cumulative gain (nDCG)
%   * mean reciprocal rank (MRR)

Another way quality of node representations can be evaluated, is using recommendation tasks. Here the idea is, instead of predicting a single item like in link-prediction, to rank the items based on their recommendation confidence. Using the ranked list, standard information retrieval metrics such as precision at rank, mean reciprocal rank and success at rank can be computed. This approach is applied to CD [@rozemberczkiGEMSECGraphEmbedding2019; @huangInformationFusionOriented2022; @faniUserCommunityDetection2020] by ranking recommendations for per community instead of on individual basis. Communities with higher scores therefore would indicate a high similarity between their members.



% @maCommunityawareDynamicNetwork2020
% 
% * Use both synthetic and real world datasets
% * Use not perse community detection baselines
% * Define auxilary helper tasks in context of *community aware* **Deep Network Embedding**:
%   * **Network Reconstruction**: Evaluates model on ability of reconstructing link structures of the network
%     * Average reconstruction precision is measured
%     * This is done for each timestamp
%     * For each node, the nearest embedding neighbors are used as predicted links


