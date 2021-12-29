# Research Questions

% The main goal of my thesis: to build a framework for community detection and representation in dynamic heterogeneous networks.
% 
% * This in order to analyse the communities within the set of datasets provided by Prof. Wang (**todo** cite work)
%   * Datasets are Dynamic, heterogeneous, Include content-based data
%   * Combination poses a problem
% * To the best of my knowledge: there are currently no algorithms that can do this
% * Direct comparison is therefore impossible
% * Introduce our research questions
%   * Elaborate a bit on their relevance
%   * In next section explain their implementation

 The main goal of my thesis is to build a framework for community detection and representation in dynamic heterogeneous networks. 

This is, to enable dynamic community analysis on the datasets proposed in @wangPublicSentimentGovernmental2020. The data described is collected from the Twitter social platform and is dynamic, heterogeneous and rich in contentual (unstructured text) data. To the best of our knowledge, there are currently no dynamic community detection algorithms that can handle this data without relaxing its rich data representation (data loss).

As there are no alike algorithms, direct comparison is not possible. To both validate merit of our methods as well as the quality of the results, we spread our research over four research questions.



[Research Question 1 (Information)]{#thm:rq1}

: *Does addition of meta-topological and/or content-based information improve quality of detected communities?*

% * Dataset rich in extra data: 
%   * meta-topological
%   * content based
% * Previous approach treat all nodes alike 
%   * ignoring most important structural features (node types)
%   * types such as hashtags can also be represented as nodes solving many issues which require topic modelling
% * Improvements in natural text analysis allow for representation of unstructured text
%   * Would this data improve quality of embeddings

While focusing on link-based features, CD algorithms treat all nodes alike. Therefore ignoring arguably most important structural features, namely node types. Additionally, supplementary node types can be constructed from categorical features enhancing network topology and solving issues requiring topic modelling. 

Similarly, recent improvements in natural text processing allow for efficient representation of natural text which is proven to improve quality of node embeddings. As formation of communities is not purely a sociological process, CD problem should benefit from incorporation of such content-based features.



[Research Question 2 (Scale)]{#thm:rq2}

: *Does usage of graph representation function learning techniques improve scale of CD beyond current state-of-the-art?*

% * In last few years new graph representation approaches were introduced
%   * Instead of operating on the whole networks
%   * They sample network using random walks or convolutions
% * As previous methods for CD ignored them (used spectral methods)
%   * Causing scalability issues posed by spectral methods
%   * It is important to test these approaches as they may yield performance improvements
% * Previous approaches use spectral methods limiting them to one network per snapshot
% * Instead learning representation function
%   * would allow limit computational complexity
%   * allow for parameter sharing across timesteps
%   * Allow for streaming graphs

Previously mentioned representation-based DCD method use spectral graph representation methods which operate on the whole network at once. More recent graph representation approaches instead learn a graph representation function by sampling the network using random walks or convolutions. 

This has a two-fold positive effects on the scalability of the algorithms. Computation can be done more efficiently as opposed to spectral methods which rely on adjacency matrices. By learning a representation function, embeddings are computed on-demand instead of being held in memory for the whole network therefore limiting impact of big networks. Other benefits may include the fact that they would be a suitable choice for processing streaming edge network variants.



[Research Question 3 (Modelling)]{#thm:rq3}

: *Does making temporal features implicit in node representations provide better quality communities as opposed to making them explicit?*

% * Previous approaches either
%   * Learn the temporal component implicitly in node representations, causing embedding be temporally aware
%   * Separate the temporal aspect explicitly by defining 

Throughout the literature various ways are used to incorporate temporal aspect into node embeddings. The implicit approach aims to make the embeddings temporally aware while the explicit approach creates a separate embedding for each snapshot. While methods using either approaches have presented good results in the literature, it is important to analyse the potential trade-off and benefits of both.



[Research Question 4 (Results)]{#thm:rq4}

: *Do community-aware node embeddings perform well on both node as well as community evaluation based tasks?*

% * While optimizing for multiple objectives (community and individual based)
%   * It is important to evaluate algorithm against other community algorithms
%   * As well as dynamic representation methods
%   * To see whether community-aware embeddings provide benefits for standard node represenation tasks

While the main task of the algorithm is to find high quality dynamic communities, the result also includes community-aware dynamic embeddings. Aside from testing the quality of the communities, it as important to compare how this community-awareness influences the embeddings on dynamic node representation tasks such as link-prediction and node classification.
