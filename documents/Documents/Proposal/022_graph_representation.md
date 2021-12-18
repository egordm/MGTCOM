## Graph Representation Learning

The representation-based approaches stem from the field of computational linguistics which relies heavily on the notion of *distributional semantics* stating that words occurring in similar contexts are semantically similar. Therefore the word representations are learned as dense low-dimensional representation vectors (embeddings) of a word in a latent similarity space by predicting words based on their context or vice versa [@mikolovEfficientEstimationWord2013; @penningtonGloveGlobalVectors2014]. Using the learned representations similarity, clustering and other analytical metrics can be computed.

The success of these representation learning approaches has spread much farther than just linguistics as similar ideas are also applied to other fields including graph representation learning. Methods such as deepwalk [@perozziDeepWalkOnlineLearning2014], LINE [@tangLINELargescaleInformation2015], and node2vec [@groverNode2vecScalableFeature2016] use random walks to sample the neighborhood/context in a graph (analogous to sentences in linguistic methods) and output vector representations (embeddings) that maximize the likelihood of preserving the topological structure of the nodes within the graph.

Whereas previously the structural information features of graph entities had to be hand-engineered, these new approaches are data-driven, save a lot of time labeling the data, and yield superior feature/representation vectors. The methods can be trained to optimize for *homophily* on label prediction or in an unsupervised manner on link prediction tasks.

Newer approaches introduce the possibility for the fusion of different data types. GraphSAGE [@hamiltonInductiveRepresentationLearning2018] and Author2Vec [@wuAuthor2VecFrameworkGenerating2020] introduce a methodology to use node and edge features during the representation learning process. Other approaches explore ways to leverage heterogeneous information present within the network by using *metapath* based random walks (path defined by a series of node/link types) [@dongMetapath2vecScalableRepresentation2017] or by representing and learning relations as translations within the embedding space [@bordesTranslatingEmbeddingsModeling2013]. In @nguyenContinuousTimeDynamicNetwork2018 the authors introduce a way to encode temporal information by adding chronological order constraints to various random walk algorithms. Other relevant advancements within the field include Graph Convolutional Networks (GCN) [@kipfSemiSupervisedClassificationGraph2017a] and (Variational) Graph Auto-Encoders (GAE) [@kipfVariationalGraphAutoEncoders2016] which present more effective ways to summarize and represent larger topological neighborhoods or whole networks.



% Various works have emerged exploring community detection using the representation learning approach. In @cavallariLearningCommunityEmbedding2017 the authors define a community as a distribution over the vector representation (embedding) space of the network (which encodes both content-based as well as topological information). Here community detection and node representation are jointly solved by defining a unified objective and alternating their optimization steps. @faniUserCommunityDetection2020 redefine node connection proximity based on learned multi-modal embedding vectors incorporating both temporal social content as well as social network neighborhood information. As *homophiliy* is optimized, more valuable communities are found within the resulting network.
% 
% * todo: 
%   * Talk in more detail about more performant algorithms which we plan to use




