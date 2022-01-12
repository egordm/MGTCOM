## Graph Representation Learning

The representation-based approaches stem from the field of computational linguistics which relies heavily on the notion of *distributional semantics* stating that words occurring in similar contexts are semantically similar. Therefore the word representations are learned as dense low-dimensional representation vectors (embeddings) of a word in a latent similarity space by predicting words based on their context or vice versa [@mikolovEfficientEstimationWord2013; @penningtonGloveGlobalVectors2014]. Using the learned representations similarity, clustering and other analytical metrics can be computed.

The success of these representation learning approaches has spread much farther than just linguistics as similar ideas are also applied to other fields including graph representation learning. Methods such as deepwalk [@perozziDeepWalkOnlineLearning2014], LINE [@tangLINELargescaleInformation2015], and node2vec [@groverNode2vecScalableFeature2016] use random walks to sample the neighborhood/context in a graph (analogous to sentences in linguistic methods) and output vector representations (embeddings) that maximize the likelihood of preserving the topological structure of the nodes within the graph.

Whereas previously the structural information features of graph entities had to be hand-engineered, these new approaches are data-driven, save a lot of time labeling the data, and yield superior feature/representation vectors. The methods can be trained to optimize for *homophily* on label prediction or in an unsupervised manner on link prediction tasks.

Newer approaches introduce the possibility for the fusion of different data types. GraphSAGE [@hamiltonInductiveRepresentationLearning2018] and Author2Vec [@wuAuthor2VecFrameworkGenerating2020] introduce a methodology to use node and edge features during the representation learning process. Other approaches explore ways to leverage heterogeneous information present within the network by using *metapath* based random walks (path defined by a series of node/link types) [@dongMetapath2vecScalableRepresentation2017] or by representing and learning relations as translations within the embedding space [@bordesTranslatingEmbeddingsModeling2013]. In @nguyenContinuousTimeDynamicNetwork2018 the authors introduce a way to encode temporal information by adding chronological order constraints to various random walk algorithms. Other relevant advancements within the field include Graph Convolutional Networks (GCN) [@kipfSemiSupervisedClassificationGraph2017a] and (Variational) Graph Auto-Encoders (GAE) [@kipfVariationalGraphAutoEncoders2016] which present more effective ways to summarize and represent larger topological neighborhoods or whole networks.



% * Goals:
%   * Introduce common graph represntation learning techniques
%   * By covering influential 



### Node2Vec

% * Learns representation vectors in graph via
%   * 2nd order random walk (node similarity depends of connected *through* node )  (dfs)
%   * 1st order random walk - learning from direct neighbors  (bfs)
% * Features bias parameter $\alpha$ that makes the bfs and dfs tradeoff
% * Random walks start from a random node and span a length of l
%   * Starting at node $v$, transition probability is calculated for each of the neighbors using
%     * 1st order (context 0): $p(u \mid v)=\frac{w(u, v)}{\sum_{u^{\prime} \in \mathcal{N} v} w\left(u^{\prime}, v\right)}=\frac{w(u, v)}{d(v)}$ 
%       * Thus prob is $1/p$ of picking a node
%     * 2nd order (context 1 node) $p(u)$ where
%       * $1/q$  probability of leaving the neighborhood
%         * where $q$ is in-out degree ratio parameter
%       * $\alpha_{p q}(t, x)= \begin{cases}\frac{1}{p} & \text { if } d_{t x}=0 \\ 1 & \text { if } d_{t x}=1 \\ \frac{1}{q} & \text { if } d_{t x}=2\end{cases}$
%       * $d_{tx}$ denotes shortest path distance between two nodes
%         * $p$ and $q$ control how fast the walk leaves a neighborhood
%   * And as such the path is chosen
% * Random walks are efficient in terms of space complexity
% * Also in time complexity since you are effectively sampling



### Graph Autoencoder

% * ...



### GraphSAGE

% * ...



### CTDN

% * ....






