## Graph Representation Learning

The representation-based approaches stem from the field of computational linguistics which relies heavily on the notion of *distributional semantics* stating that words occurring in similar contexts are semantically similar. Therefore the word representations are learned as dense low-dimensional representation vectors (embeddings) of a word in a latent similarity space by predicting words based on their context or vice versa [@mikolovEfficientEstimationWord2013; @penningtonGloveGlobalVectors2014]. Using the learned representations similarity, clustering and other analytical metrics can be computed.

The success of these representation learning approaches has spread much farther than just linguistics as similar ideas are also applied to other fields including graph representation learning. Methods such as Deepwalk [@perozziDeepWalkOnlineLearning2014], LINE [@tangLINELargescaleInformation2015], and node2vec [@groverNode2vecScalableFeature2016] uses random walks to sample the neighborhood/context in a graph (analogous to sentences in linguistic methods) and output vector representations (embeddings) that maximize the likelihood of preserving the topological structure of the nodes within the graph.

Whereas previously the structural information features of graph entities had to be hand-engineered, these new approaches are data-driven, save a lot of time labeling the data, and yield superior feature/representation vectors. The methods can be trained to optimize for *homophily* on label prediction or in an unsupervised manner on link prediction tasks.

Newer approaches introduce the possibility for the fusion of different data types. GraphSAGE [@hamiltonInductiveRepresentationLearning2018] and Author2Vec [@wuAuthor2VecFrameworkGenerating2020] introduce a methodology to use node and edge features during the representation learning process. Other approaches explore ways to leverage heterogeneous information present within the network by using *metapath* based random walks (path defined by a series of node/link types) [@dongMetapath2vecScalableRepresentation2017] or by representing and learning relations as translations within the embedding space [@bordesTranslatingEmbeddingsModeling2013]. In @nguyenContinuousTimeDynamicNetwork2018 the authors introduce a way to encode temporal information by adding chronological order constraints to various random walk algorithms. Other relevant advancements within the field include Graph Convolutional Networks (GCN) [@kipfSemiSupervisedClassificationGraph2017a] and (Variational) Graph Auto-Encoders (GAE) [@kipfVariationalGraphAutoEncoders2016] which present more effective ways to summarize and represent larger topological neighborhoods or whole networks.



% * Goals:
%   * Introduce common graph representation learning techniques
%   * By covering influential 

In the remainder of this section, we briefly describe the working of a selection of influential representation learning algorithms. 



### Negative Sampling

Negative sampling is a technique used for reducing the calculation complexity of loss for link prediction tasks. Originally introduced in @mikolovDistributedRepresentationsWords2013 this technique was used to optimize the *skip-gram* algorithm which predicts context words based on a single input word (See +@eq:skipgram). Computing the result of this softmax function is very expensive as the vocabulary may become very large and the negative samples outnumber the positive ones by a lot. Negative sampling (+@eq:negativesampling) is introduced as an approximation where $w_O$ is the output context word, given the $w_I$ the input word, $v_{w_O}, v_{w_I}$ as their representation vectors respectively, sigmoid function $\sigma$, and a sample of $n$ words sampled from a random distribution weighted by word frequency $P_n(w)$.

Due to its efficiency, negative sampling is also embraced in graph representation learning as the link prediction task is synonymous with context word prediction.

$$
p\left(w_{O} \mid w_{I}\right)=\frac{\exp \left(v_{w_{O}}^{\prime}{ }^{\top} v_{w_{I}}\right)}{\sum_{w=1}^{W} \exp \left(v_{w}^{\prime} v_{w_{I}}\right)} 
$$ {#eq:skipgram}

$$
\log P(w_O|w_I) \approx \log \sigma\left(v_{w_{O}}^{\prime}{ }^{\top} v_{w_{I}}\right)+\sum_{i=1}^{k} \mathbb{E}_{w_{i} \sim P_{n}(w)}\left[\log \sigma\left(-v_{w_{i}}^{\prime}{ }^{\top} v_{w_{I}}\right)\right]
$$ {#eq:negativesampling}

### Node2Vec

% * @groverNode2vecScalableFeature2016
%   Learns representation vectors in graph via
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
% * Similar to language modelling, negative sampling is used for loss
% * Random walks are efficient in terms of space complexity
% * Also in time complexity since you are effectively sampling
% * @wuAuthor2VecFrameworkGenerating2020
%   * introduces an extension that maps different types of node / edge embeddings to a common space
%   * (for heterogenous networks)

Introduced in @groverNode2vecScalableFeature2016, node2vec learns node representations within a graph by introducing a hybrid *second-order* random walk technique. Once random walks are constructed, the *skip-gram* approach along with negative sampling is used to learn the node representations. Since random walks can be efficiently sampled, this helps to avoid memory and runtime complexity issues faced by formerly leading approaches such as spectral/matrix factorization methods.

The random walks start from a random node $u$ and span a length of $k$ nodes. The next node in the walk is chosen by calculating the transition probability for each of the neighbors of the current node. See +@eq:randomwalk1o for calculation of transition probability for first-order random walks where next node $u$ is chosen given only the current node $v$ and the corresponding edge weights $w(u, v)$. 

$$
p(u \mid v)=\frac{w(u, v)}{\sum_{u^{\prime} \in \mathcal{N} v} w\left(u^{\prime}, v\right)}
$$ {#eq:randomwalk1o}

Second-order random walk (+@eq:randomwalk2o) instead considers last two visited nodes $v, t$ and introduces an additional tradeoff weight $\alpha_{pq}(t,x)$ parametrized by $p$ (*in-out bias*) and $q$ (*return ratio*). See +@eq:node2vectradeoff for the trade-off weight where $d_{tx}$ represents the hop distance between two nodes $t$ and $x$. Given the distance of the two nodes the first-order random walk is therefore additionally biased towards two cases: (i) returning to the same node $d_{tx} = 0$, therefore reinforcing exploration of a single neighborhood (BFS), (ii) exiting the neighborhood of a single node (DFS) (See @fig:randomwalk). Their experiments have shown that setting a higher $q$ parameter to optimize embeddings for *homophily* and yield better clustering results while setting a higher $p$ value promotes structural equivalence which aids in node classification tasks.

$$
\alpha_{p q}(t, x)= \begin{cases}\frac{1}{p} & \text { if } d_{t x}=0 \\ 1 & \text { if } d_{t x}=1 \\ \frac{1}{q} & \text { if } d_{t x}=2\end{cases}
$$ {#eq:node2vectradeoff}

$$
p(u \mid v, t)=\frac{\alpha_{p q}(t, u) w(u, v)}{\sum_{u^{\prime} \in \mathcal{N} v} \alpha_{p q}\left(t, u^{\prime}\right) w\left(u^{\prime}, v\right)}
$$ {#eq:randomwalk2o}

![BFS and DFS search strategies from node $u$ ($k = 3$).](/home/egordm/.config/marktext/images/7488500d2d1a170449bad6632dee81276b2ab8bd.png){#fig:randomwalk width=250px}



### Graph Autoencoder (GAE)

% * Applies the idea of Variational Autoencoders to the graph structure
% * Traditionally autoencoders consist of
%   * An encoder encoding the input $X$ to a low dimensional latent space
%   * A decoder decoding the low dimensional vector back to the input space $\hat{X}$
% * Reconstruction error is calculated $L(X, \hat{X}) = ||X - \hat{X}||^2$
% * Provides robustness as input can be easily augmented and changes can be measured
%   * Variational autoencoders produce a distribution instead (to be able to generate new samples)
% * Their architecture is similar to spectral GCN which work on
%   * Whole graph Laplacian matrix - which provides a mathematical representation of the graph (often in a normalized way)
%   * Decoder reconstructs this Laplacian matrix by calculating the inner product of the latent matrices (stack of low-dim repr vectors)

In @kipfVariationalGraphAutoEncoders2016 authors describe the application of the autoencoder concept for graph representation learning. Traditionally autoencoders consist of an encoder $E: X \rightarrow Z$ translating the input $X$ to a low dimensional latent space vector $Z$, and a decoder $D: Z \rightarrow \hat{X}$ translating the low dimensional vector $Z$ back to the input space as $\hat{X}$. To optimize both components *reconstruction error* $L(X, \hat{X}) = ||X - \hat{X}||^2$ and its variants are employed. This ensures cohesion of the latent space.

The architecture for graph autoencoders is similar to the one used in spectral graph convolution networks. The whole graph is represented as a graph Laplacian matrix $S \in \mathbb{R}^{N\times N}$ (mathematical representation of a graph; can be adjacency, similarity matrix, or similar). During encoding, an efficient factorization approximation of this matrix is produced as $Z \in \mathbb{R}^{N \times d}$, while during decoding the inner product is calculated $ZZ^T$ reconstructing the graph Laplacian.

Employing this approach for graph representation learning yields many benefits such as flexibility for objective definition and robustness against noise. The main downside of this approach is that working on a full graph is expensive and doesn't scale well.



![Graph Autoencoder architecture](/home/egordm/.config/marktext/images/d1d5cd4d01bc507a198256ec0a4e2b9982e6a10b.png){width=380px}



### GraphSAGE

% * @hamiltonInductiveRepresentationLearning2018
%   Was made to tackle issues in existing approaches are *transductive*  
%   * ie. needs the whole graph to learn to embed a node
%   * generalize poorly since the addition of a node requires a rerun of the algorithm
% * Representation function learning method - referred to as aggregator functions
%   * These functions can be used to predict the embedding rather than statically learning it
% * Needs a little bit different sampling approach than random walks
%   * Due to the fact aggregator functions need to be stacked (which need regular input counts)
%   * Therefore a neighborhood of a node is sampled to the depth $k$ (with $k$ being the number of aggregator functions)
% * Works by initializing all node embeddings to node features
%   * For each node its embedding is concatenated with the aggregated representation of its neighbors
%   * And passed to a deep neural network yielding a representation vector at a given layer
% * Loss is defined similarly using negative sampling
% * TODO: Talk about PinSAGE extension
%   * Introduces a methodology for more efficient sampling and importance weighing
%   * Introduces a way to learn representations of heterogeneous networks
%     * By using type-specific aggregation functions mostly for metapaths

Existing graph representation learning approaches are *transductive*, meaning the algorithm needs the whole graph to learn the embedding of a node. This approach generalizes poorly since the addition of a single node requires a rerun of the algorithm. In @hamiltonInductiveRepresentationLearning2018 the authors introduce the GraphSAGE representation function learning method which solves this issue by learning a set of aggregator functions to predict the embedding rather than statically learning it (also referred to as *inductive learning*).

First, the layer count parameter $k$ is specified defining the number of aggregations used, and therefore the aggregation neighborhood size (namely $k$-hop neighborhoods). A forward pass for the "mean" operation-based aggregator is defined as +@#eq:graphsageagg. Where $h^{k-1}_v$ is the representation vector of a node $v$ at $k-1$-th layer, $\mathbf{W}$ is the the set of weights, and $\sigma$ is the activation function. By initializing $h^0_v$ using the node's feature vector and applying aggregator functions recursively given the neighborhood of a node $\mathcal{N}(v)$ the final representation vector $h^k_v$ is calculated (See +@fig:graphsage).

In follow-up work @yingGraphConvolutionalNeural2018 the authors introduce a random walk simulating sampling-based aggregation approach which in combination with negative sampling is capable of learning on web-scale graphs (exceeding > billion nodes).  

$$
\mathbf{h}_{v}^{k} \leftarrow \sigma\left(\mathbf{W} \cdot \operatorname{MEAN}\left(\left\{\mathbf{h}_{v}^{k-1}\right\} \cup\left\{\mathbf{h}_{u}^{k-1}, \forall u \in \mathcal{N}(v)\right\}\right)\right.
$$ {#eq:graphsageagg}

![Overview of GraphSAGE model architecture using depth-2 convolutions ($k=2)](/home/egordm/.config/marktext/images/f844cf41b511a806c4af337a2c48db8da70b89ac.png){#fig:graphsage width=380px}
