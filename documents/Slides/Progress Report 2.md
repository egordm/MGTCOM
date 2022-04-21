# MyGreaTCOM: Multimodal Graph Temporal Community Detection Framework

## Addressing Issues

* Dynamic Number of Communities
* One-hop positive sampling learns wrong embeddings (direct neighborhood may be more complext - See DBLPHCN)
* Can't rely on existence of node features
  
  

## Community Definition

* Differs per task
* Combination of patterns in:
  * Content
  * Topology
  * Metatopology
  * Temporality

<img src="file:///home/egordm/.config/marktext/images/2022-04-20-14-33-49-image.png" title="" alt="" width="584">

* Assumptions:
  * Homophiliy: Nodes connected to each other (in topology) are similar **in content**
  * Temporal Homophily: Nodes connected to each other (in topology) are similar **in time**
* Want:
  * A community detection which can account/select both
    
    Provide clustering with all the information
    
    and let it find the patterns in the data
    
    

## TopoNet - Learning on Homophily

* Standard Node2Vec already performs well but has a few problems:
  * Doesnt taske metatopology into consideration
  * Can't deal with unseen / untrained nodes

<img src="file:///home/egordm/.config/marktext/images/2022-04-20-14-43-39-image.png" title="" alt="" width="414">

* HGT or GraphSAGE work well (Graph Convolutions)
  * Can't deal with nodes that have no features
  * Training then in unsupervised manner is not out of the box



<img src="file:///home/egordm/.config/marktext/images/2022-03-09-10-12-47-image.png" title="" alt="" width="504">



* Solution:
  * Use Conv for feature extraction $V_{feat}$
  * Use N2V random walks for feature learning $V_{topo}$
  * Embed nodes with no` `features
    * When adding new node infer features while setting own feature vector to 0
* <img src="file:///home/egordm/.config/marktext/images/2022-04-20-14-42-44-image.png" title="" alt="" width="560">

* Properties:
  * Feature extraction is trained on homophily
  * Embeddings are trained on feature extraction
  * When a new node is introduced:
    * Is fed to the feature extraction:
      * With available feature vector
  * With zero feature vector (feature vector is inferred based on the neighborhood)
  * * Loss: $L_{topo}$
      
      

## TempoNet - Learning on Temporal Homophily

* How do we learn temporal relation between nodes?
* What if a node does not have a timestamp?
  
  

* Naive approach:
  * Do temporal random walk:
    * Start at a node
    * Define a time window
    * Walk to nodes present in same timewindow
  * Problems:
    * What if a starting node does not have a timestamp?
    * Topological bias - temporally related nodes may not be connected
    * Dead ends - restart?
* TODO: Visualize random walk
  
  

* Introducing Ballroom Sampling
  * Unbiased Temporal sampling
  * Flow:
    * Pick a small enough time window - based on dataset statistics
    * Pick a node
    * If a node has no timestamp:
      * Do a temporal random walk
      * Stop when you come across a node with a timestamp - use this timestamp
    * ROI: timestamp + window
    * Pick #walks_per_node temporal neighbors
      * neighbors co-occuring in the same temporal ROI anywhere in graph
    * For each neighbor do a temporal random walk (staying within ROI)
      * When deadend occurs - restart random point
    * Collect context:
      * All the occurred nodes along #walks_per_node walks of #walklength long
    * Return: random #k nodes from the context
  * Efficiency:
    * Note how collected context is relevant for all nodes in the context
    * -> Sampe context multiple times for different neighbors
  * Samples

<img title="" src="file:///home/egordm/.config/marktext/images/2022-04-20-14-41-18-image.png" alt="" width="559">

<img src="file:///home/egordm/.config/marktext/images/2022-04-20-14-41-47-image.png" title="" alt="" width="416">

* Loss: $L_{tempo}$
* Result: $V_{tempo}$

## Combined Network

* Reused feature extraction $V_{feat}$ 
* Train TopoNet ($V_{topo}$) and TempoNet ($V_{tempo}$) in ensemble
  * Batch by same central nodes
  * Sampled neighbors may differ but are not relevant
* Loss: $L_{tempo} + L_{topo}$
* Output: either $V_{topo} || V_{tempo}$ or $V_{topo} + V_{tempo}$  - determine wheter there is a difference in ablations
  
  

## Clustering

* Definitions:
  * $x_i \in \mathbb{R}^{D}$: feature vector for data point i
  * $r_i \in \{0.0..1.0\}^{K}$ : soft assignment of datapoint i 
  * $z_i \in \{0..K\}$ : hard assignment of data point i
  * $\mu_k \in \mathbb{R}^{D}$ : cluster center k
  * $K \in \mathbb{N}$ : Cluster count
* Formerly:
  * Fit cluster centers and compute isotropic cluster loss
  * ![](/home/egordm/.config/marktext/images/2022-04-20-13-11-05-image.png)
  * Problems:
    * Clusters may become empty or oversaturated
    * Assumes clusters are spherical (in 3d) - imagine that for $D$ dimensions
    * No way to add more clusters
      
      

## Paper discussion: Chang and Fisher III

J. Chang and J. W. Fisher III, “Parallel Sampling of DP Mixture Models using Sub-Cluster Splits,” in Advances in Neural Information Processing Systems, 2013, vol. 26. Accessed: Apr. 09, 2022. [Online]. Available: [Parallel Sampling of DP Mixture Models using Sub-Cluster Splits](https://papers.nips.cc/paper/2013/hash/bca82e41ee7b0833588399b1fcd177c7-Abstract.html)



* **Parametric Model**: amount of clusters if not known
* **Non Parametric Model**: amount of cluster is known
  
  

#### Parametric Models: Gaussian Mixture Models

* Gaussian Mixture Model is defined by:
  * $\mu_k$: Cluster mean
  * $\Sigma_k$ : Cluster covariance (in most cases a diagonal matrix)
  * $\pi_k$ Cluster saturation (or proportion of dataset covered by the cluster)
  * $\theta_k = \{\mu_k, \Sigma_k\}$ cluster parameters
* Given $K$ find $\theta$
  
  

![scikit learn  How to evaluate the loss on a Gaussian Mixture Model?   Cross Validated](https://i.stack.imgur.com/s3QiK.png)

Image above: Covers a single feature

![21 Gaussian mixture models — scikitlearn 102 documentation](https://scikit-learn.org/stable/_images/sphx_glr_plot_gmm_covariances_001.png)

Image above: Covers a 2 features and varies the type of covariance



* Usually done using Expectation Maximization (EM)
  * An amoritized way to do max likelihood estimation
  * Flow:
    * Initialize $\theta$
    * Expectation: Assign points to clusters yielding $z$
    * Maximization: Update $\theta$ based on $z$ and $X$
  * Properties:
    * Guaranteed to converge
    * Very cheap
    * Can be computed incrementally
    * Assignment can be done using cluster centers or a model
    * May converge at local optimum

![Expectation–maximization algorithm - Wikipedia](https://upload.wikimedia.org/wikipedia/commons/6/69/EM_Clustering_of_Old_Faithful_data.gif)



### Non Parametric Models: Dirichlet Process Mixture Models (DPMM)

* Dirichlet Process defines a way to quantize the probability a distribution comes from another distribution
  * Example:
    * Given dataset distribution: $\mathcal{D}$
    * Give probability that distributions: $D_1$ and $D_2$ come from it
      
      

![](/home/egordm/.config/marktext/images/2022-04-20-13-43-32-image.png)



* Therefore:
  * Infer initial dataset distribution $\theta_D$
  * Now you can check for each cluster, the likelihood $\theta_k \sim \theta_D$
    
    

* Defining clustering task as a Markov Chain:
  * Splitting / merging clusters can be seen as a transition between different states
  * But what is the probability??

<img src="file:///home/egordm/.config/marktext/images/2022-04-20-13-48-09-image.png" title="" alt="" width="455">

* Metropolis Hastings MCMC
  * Method for obtaining a sequence of [random samples](https://en.wikipedia.org/wiki/Pseudo-random_number_sampling "Pseudo-random number sampling") from a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution "Probability distribution") from which direct sampling is difficult. This sequence can be used to approximate the distribution - wikipedia
  * ![](/home/egordm/.config/marktext/images/2022-04-20-13-52-56-image.png)
  * $x'$ is the new state
  * A(x', x) is the acceptance ratio of a transition
  * Defined by balance: ![](/home/egordm/.config/marktext/images/2022-04-20-13-53-51-image.png)
* Thus the cluster split probability:
  * ![](/home/egordm/.config/marktext/images/2022-04-20-13-54-28-image.png)
  * $\lambda$ are the model parameters ($\pi, \theta$)
  * $\alpha$ is a prior parameter shifting the probability of a split vs merge
    
    

#### But how are subclusters defined?

* Just the the nested version of the DPMM
  * Clusters $\theta_k$ are distributions from the full dataset distrubution $\theta_D$
  * Subclusters $\theta_{ki}$ are distributions from the clusters $\theta_k$
  * Therefore you can do EM over clusters and subclusters in ensamble to find their parameters
    * Caveat: so long as clusters dont do too large shifts so subclusters can keep up
      
      

* Priors
  * There may be some assumptions you want to make over the dataset:
    * Dirichlet prior - is for categorical things like cluster distribution
    * But we also need a prior for $\theta$ (both dataset and cluster params) - AKA Conjugate priors
      * Since they can not perfectly fit the data, bit of fuzziness needs to be preserved
  * Unrelated example:
    * Knowing the liverpool won last two games, what is the probability that they win the next one?
      * It is not 1
  * **Conjugate Prior**: $p(x|\theta)$
    * conjugate prior can generally be determined by inspection of the [probability density](https://en.wikipedia.org/wiki/Probability_density_function "Probability density function") or [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function "Probability mass function") of a distribution - wikipedia
  * These priors are used to weigh the DPMM parameters to preserve the fuzziness
    
    

### Paper Discussion: Ronen et al.

M. Ronen, S. E. Finder, and O. Freifeld, “DeepDPM: Deep Clustering With an Unknown Number of Clusters,” arXiv:2203.14309 [cs, stat], Mar. 2022, Accessed: Apr. 09, 2022. [Online]. Available: [[2203.14309] DeepDPM: Deep Clustering With an Unknown Number of Clusters](http://arxiv.org/abs/2203.14309)



* Use Normal Inverted Wishart prior
  * Is easy to incrementally update
* They define a deep neural net for estimating $r_i$ (soft assignments)
  * Neural net tries to catchup by apprixmating $\theta$
  * While $\theta$ is updated using neural net assignments
* Result is $\theta$ and a classifier
  * Loss is the negative likelihood of a point being assigned to a cluster  $L_{clus}$
    
    

## MGTCOM End2End

* Combines the feature learning and clustering pipelines
* Works by alternating between
  * Learning features based on:
    * $L_{tempo} + L_{topo} + L_{clust}$ (the losses are weighted)
  * Updating $\theta$ to fit the features
    
    

## Demo

* DPMM SC - STAR WARS = [Weights & Biases](https://wandb.ai/egordm/ThesisDebug/runs/3ul4ff8d?workspace=user-egordm)
* DPMM SC - synthetic = [Weights & Biases](https://wandb.ai/egordm/ThesisDebug/runs/i62trrmj?workspace=user-egordm)
* Het2Vec IMDB5000 =
  * [Weights & Biases](https://wandb.ai/egordm/ThesisDebug/runs/5dkmkl89)
  * /data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/Het2VecExecutor/IMDB5000/wandb/run-20220420_124041-5dkmkl89/files/graph.graphml 
* Het2Vec DBLP = 
  * /data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/Het2VecExecutor/DBLPHCN/wandb/run-20220420_124912-19z7ywcz/files/graph.graphml/
  * [Weights & Biases](https://wandb.ai/egordm/ThesisDebug/runs/19z7ywcz?workspace=user-egordm)
* ICEWS = 
  * /data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/Het2VecExecutor/ICEWS0515/wandb/run-20220420_130115-19vyihmc/files/
  * [Weights & Biases](https://wandb.ai/egordm/ThesisDebug/runs/19vyihmc?workspace=user-egordm)





## Whats next

* Fix current issues:
  * Metric space: Dot product works very well but is very very bad at visualization
  * Euclidean space performs very very poorly in node2vec
* Finish the E2E implementation
* Collect results
  * Clustering versus modularity
  * Clustering versus clustering metrics
  * Same but per snaphot - without retraining
  * Collect results for held out nodes - inference   (part og RQ2)
* Ablations:
  * TempoNet only
  * TopoNet only  (RQ1)
  * Combined net - then cluster separately
  * topo cat tempo vs topo plus tempo -- (RQ3)
  * E2E model 
    * incorporate L_clus vs not  (RQ4)
    * 
* Write everything down in the thesis (formally )
* 
