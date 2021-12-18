## Representation-based Approaches

### Community Detection

#### Affiliation Graph Networks



% * @kangCommunityReinforcementEffective2021
%   
%   * Present a **Community Reinforcement** approach
%     
%     * Is CD algorithm agnostic
%       
%       * Shown in experiments - therefore the graph itself benefits
%     
%     * Reinforces the Graph by
%       
%       * Creating inter-community edges
%       
%       * Deleting intra-community edges
%       
%       * Determines the appropriate amount of reinforcement a graph needs
%     
%     * Which helps the dense inter-community and sparse intra-community property of the graph
%       
%       * Can effectively turn difficult-to-detect community situation into that of easy-to-detect communities
%     
%     * Challenges:
%       
%       * Needs to be unsupervised (doesn't need community annotations to work)
%       
%       * Appropriate amount of reinforcement needs to be determined (otherwise noise is introduced)
%       
%       * Needs to be fast, checking every possible edge is infeasible
%     
%     * Methodology:
%       
%       * Edge Addition / Deletion
%         
%         * Based on node similarity of connected pairs
%           
%           * Similar nodes are likely to be in a community (intra edges)
%           
%           * Dissimilar ones are less likely to be in a community (inter edges)
%         
%         * Employ graph embedding to generate topological embeddings
%           
%           * Adamic/Adar (local link-based)
%           
%           * SimRank (gloabl link-based)
%           
%           * Node2vec graph embedding based
%         
%         * Predict similarities and bucket them
%           
%           * Use similarity buckets to select intra and inter edges - and to tune the model
%           
%           * Buckets are selected on how well they predict current edges
%       
%       * Detect the right amount of addition
%         
%         * Use a gradual reinforcement strategy by generating a series of graphs (adding top x inter and removing intra edges)
%         
%         * Pick the best graph using a scoring function (modularity)
%         
%         * Simply run CD over the graph and see
%       
%       * Reducing comutational overhead
%         
%         * Using a greedy similarity computation
%         
%         * Prefer nodes which are likely to be in same community of inter similarity detection
%     
%     * Tests results on:
%       
%       * Synthetic Graphs: LFR
%       
%       * Real world graphs: Cora, Siteseer, DBLP, Email



% * @huangInformationFusionOriented2022
%   
%   * Their own made dataset (rumor detection): [GitHub - MingqingHuang-SHU/HRTCD: Information Fusion Oriented Heterogeneous Social Network for Friend Recommendation via Community Detection](https://github.com/MingqingHuang-SHU/HRTCD)
%   
%   * Recommendation of friends based on per user detected communities
%   
%   * Communities are detected on per user contructed subnetworks.
%   
%   * Approach supports multiple communities per user, but is not one of global community detection



% * @luberCommunityDetectionHashtagGraphsSemiSupervised2021
%   
%   * Not really focused on community detection
%   
%   * More focused on enhancing topic modelling using community detection
%   
%   * Communities are from Hashtag graphs which help encode structural / content-based information of the tweets in context of other tweets



% * @rozemberczkiGEMSECGraphEmbedding2019
%   
%   * Learns clustering (centers) simultaneously with computing embeddings
%   
%   * Objective functions includes:
%     
%     * Term to embed around the origin
%     
%     * Term to force nodes with similar sampled neighborhoods to be embedded close
%     
%     * Term to force nodes to be close to the nearest cluster (weighted by clustering coefficient)
%   
%   * Weights are randomly initialized
%   
%   * Clustering coefficient is annealed (changes overtime)
%   
%   * Uses negative sampling to avoid huge cost of softmax
%   
%   * Adds regularizer "social network cost" to optimize for homophiliy
%     
%     * weighs distance in latent space by neighborhood overlap
%     
%     * Makes algorithm more robust to changes in hyperparameters
%   
%   * Evaluate cluster quality by modularity
%   
%   * Evaluate embeddings by genre prediction / recommendation



% * @jiaCommunityGANCommunityDetection2019
%   
%   * Has some info in related work to extend on graph representation learning
%   
%   * Solves issue of detecting overlapping communities:
%     
%     * K-means and Gaussian Mixture Model cant do that
%   
%   * Proposes CommunityGAN:
%     
%     * Solves graph representation learning and community detection jointly
%     
%     * Embeddings indicate the membership strength of vertices to communities
%     
%     * Make use of Affiliation Graph Model AGM for community (detection) assignment
%     
%     * Uses GAN structure to:
%       
%       * Generate most likely vertext subset s to compose a specified kind of motif
%         
%         * "Graph AGM" motif generation model
%       
%       * Discriminate whether vertex subset s is a real motif or not
%       
%       * Motif = in this case is an n-clique
%     
%     * Study motif distribution among ground truth communities to analyse how they impact quality of detected communities
%   
%   * Methodology:
%     
%     * Define a method to efficiently random walk such cliques / motifs
%     
%     * Generator tries to learn p_{true}(m|v_c) as preference distribution of motifs containing v_c vs all the motifs
%       
%       * To be able to generate most likely motifs (vertex subsets) similar to real motifs covering v_c
%     
%     * Discriminator tries to learn probability of a vertex subset being a real motif
%       
%       * Tries to discriminate between ground truth motifs and not
%     
%     * AGM:
%       
%       * Can define a measure to check whether two nodes are affiliated through a specific (or any) community
%       
%       * Usually applied for edge generation
%       
%       * In this case, extended to motif generation (edge is a 2-clique)
%         
%         * The affiliation is defined now in form of a motif in community
%     
%     * Amount of communities are chosen by hyperparameter tuning



% * @yangGraphClusteringDynamic2017
%   
%   * Goal: unsupervised clustering on networks with contents
%     
%     * Propose a way to utilize deep embedding for graph clustering
%   
%   * Simultaneously solve node representation problem and find optimal clustering in a e2e manner
%     
%     * Jointly learns embeddings X and soft clustering q_i \in Q
%     
%     * \sum_k q_{ik}: probablility of node v_i belonging to kth cluster
%     
%     * K is known a-priori
%   
%   * Employ Deep Denoise Autoencoder (DAE) - good for features with high-dimensional sparse noisy inputs
%   
%   * Use stable influence propagation technique (for computing embeddings)
%     
%     * Use a transition matrix for a single step embedding propagation
%     
%     * Because:
%       
%       * Random walk requires more tuning
%       
%       * Their transition matrix is very similar to a spectral method (symmetric Laplacian matrix)
%       
%       * Influence propagation is like kipf and welling - doenst require matrix decomposition
%     
%     * Embedding loss: \mathcal{J}*{1}=\sum*{i=1}^{n} l\left(\mathrm{a}*{i}, \tilde{\mathrm{a}}*{i}\right)
%   
%   * Introduce GRACE cluster module:
%     
%     * Computes soft clustering Q from: q_{i k}=\frac{\left(1+\left\|\mathbf{x}_{i}-\mathbf{u}_{k}\right\|^{2}\right)^{-1}}{\sum_{j}\left(1+\left\|\mathbf{x}_{i}-\mathbf{u}_{j}\right\|^{2}\right)^{-1}}
%     
%     * Learn clustering results by learning distribuition P where p_{i k}=\frac{q_{i k}^{2} / f_{k}}{\sum_{j} q_{i j}^{2} / f_{j}}
%       
%       * and f_{k}=\sum_{i} q_{i k} total number of nodes softly assigned to kth cluster
%     
%     * Clustering Loss: \mathcal{J}_{2}=K L(\mathcal{P} \| Q)=\sum_{i} \sum_{k} p_{i k} \log \frac{p_{i k}}{q_{i k}}
%     
%     * Training is done in alternating steps:
%       
%       * Macrostep: Compute: P and fix it
%       
%       * S Microsteps: Update node embeddings S and cluster centers U
%         
%         * Tries to make Q catch up with P



### Dynamic Community Detection



% * @faniUserCommunityDetection2020
%   
%   * Propose a new method of identifying user communities through multimodal feature learning:
%     
%     * learn user embeddings based on their **temporal content similarity**
%       
%       * Base on topics of interest
%       
%       * Users are considered like-minded if they are interested in similar topics at similar times
%       
%       * Learn embeddings using a context modelling approach
%     
%     * learn user embeddings based on their **social network connections**
%       
%       * Use GNN which works as a skip-gram like approach by generating context using random walks
%     
%     * **interpolate** temporal content-based embeddings and social link-based embeddings
%   
%   * Then they use these multimodal embeddings to detect dynamic communities\
%     
%     * Communities are detected on a modified graph
%       
%       * Weights are set given embedding similarity
%       
%       * Communities are detected using louvain methods
%     
%     * Then test their approach on specific tasks such as
%     
%     * News recommendation
%     
%     * User for content prediction
%   
%   * Note: **This approach detects static communities**
%     
%     * But the communities implicitly take time into account



% * @wangVehicleTrajectoryClustering2020
%   
%   * Transform task of trajectory clustering into one of Dynamic Community Detection
%     
%     * discretion the trajectories by recording entity their current neigbors at each time interval
%     
%     * Edge streaming network is created
%   
%   * Use representation learning to learn node their embeddings
%     
%     * Use dyn walks to perform random walks in time dimenstion
%     
%     * Use negative sampling to avoid the softmax cost
%   
%   * Then use K-means to find the communities
%     
%     * Try K-means, K-medioids and GMM (Gaussian Mixture Models)
%     
%     * Initalize the centers at the previous timestamp centers
%   
%   * Use quality measures to establish quality of results



% * @wangEvolutionaryAutoencoderDynamic2020
%   
%   * Approach is similar to to @maCommunityawareDynamicNetwork2020
%   
%   * Defines a unified objective where
%     
%     * community characteristics
%     
%     * previous clustering
%     
%     * are incorporated as a regularization term
%   
%   * **They argue that real world networks are non-linear** in nature and **classical approaches can not capture this**
%     
%     * Autoencoders can though
%   
%   * Methodology:
%     
%     * Construct a similarity matrix using Dice Coefficient (handles varying degrees well)
%     
%     * Apply stacked (deep) autoencoders to learn the low-dimensional representation
%     
%     * Characterizes tradeoff between two costs:
%       
%       * Snapshot cost (SC):
%         
%         * Reconstruction loss
%         
%         * Node embedding similarity (homophiliy) between connected nodes
%         
%         * Node embedding similarity (homophiliy) between nodes in same community
%       
%       * Temporal cost (TC)
%         
%         * Temporal smoothness of node embeddings
%     
%     * Adopt K-means to discover community structures



% * @maCommunityawareDynamicNetwork2020 (use as baseline?)
%   
%   * Define communities in terms of large and small scale communities
%   
%   * They propose a method for dynamic *community aware* network representation learning
%     
%     * By creating a unified objective optimizing stability of communities, temporal stability and structure representation
%     
%     * Uses both first-order as well as second order proximity for node representation learning
%   
%   * They define community representations as average of their members
%     
%     * Adopt a stacked autoencoder to learn low-dimensional (generic) representations of nodes
%   
%   * They define loss in terms of:
%     
%     * Reconstruction Error: How well the graph can be reconstructed from the representation
%     
%     * Local structure preservation: Preservation of homophiliy - connected nodes are similar
%     
%     * Community evolution preservation: Preservation of smoothness of communities in time at multiple granularity levels
%   
%   * The communities are initialized using classical methods:
%     
%     * First large communities are detected using Genlouvin (fast and doesnt require priors)
%     
%     * Then small scale communities are detected using k-means by defining a max community size w
%       
%       * Which provides more fine tuned communities
%   
%   * Using the initial embeddings the temporal embeddings are optimized
%     
%     * Done by optimizing all at once - therefore maintaining the stability
%     
%     * And use of the mentioned combined objective
%   
%   * Though they present / evaluate their algorithm in terms of Dynamic Representation Algorithms
%     
%     * Therefore the actual quality of communities remains to be known