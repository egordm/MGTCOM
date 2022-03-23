# Changelog
All notable changes to this project will be documented in this file.

## [2022.12] - 2022-03-23

### Bug Fixes

- Fixed kmeans initialization regarding inferring the embedding size
- Fixed louvain initialization normalization; Added support for configurable distance functions

### Features

- Updated changelog and updated push command to include tags
- Updated star wars dataset preprocessing scripts
- Updated todo list and progress report
- Added support for temporal HGT
- Updated DBLP and IMDB preprocessing to include binary features about the nodes
- Added dblp, imdb and star-wars torch_geometric dataset loading scripts
- Added HeteroEdgeDataset and HeteroNodeDataset. Also added a contrastive sampling based dataloader
- Added helper functions to partition hetero data and to extract unique nodes from graphs
- Added separate clustering and embedding modules
- Minor refactors regarding logging and graph loading
- Added separate hinge loss module. Added modularity calculation function based on heterodata graph storage
- Added positional model for positional community aware clustering
- Added example training script using Torch Lightning for community aware embeddings
- Added datamodule combining the positive sampling and node batching for prediction
- Updated training script to use IMDB and DBLPHCN dataset
- Added cluster initialization
- Added cleaner metric summarization in progress bar. Added cosine lr annealing
- Added to undirected graph transformation *without edge aggregation*
- Added temporal contrastive window based sampling dataloader
- Added cli for positional training / encoding
- Added temporal training script
- Tested temporal models
- Updated social distancing student dataset with feature engineering
- Added social distancing student dataset

### Refactor

- Moved old embedding and dataset classes into experiments to maintain compatibility

### Testing

- Added dblp, imdb and star-wars dataset tests

## [2022.10] - 2022-03-09

### Features

- Added normalization parameter for node embedding learning
- Updated activate script to work in non-root folders
- Temporarily changed cudatoolkit to dev version. Should change back once tch-geometric is built by ci
- Added pinsage example aka graphsage with neighbor importance weighing
- Added pinsage and tch geometric based temporal community detection
- Added train-tch utilizing negative sampling and neighbor sampling from tch_geometric
- Added proof of concept heterogenous graph learning / clustering
- Updated tch hetero to use aggregation functions. Cleaned up loss function
- Added progress report

## [2022.08] - 2022-02-18

### Bug Fixes

- Updated report graphs to show metric name as title
- Fixed visualization export to swap out edgelist extension for graphml (new standard format)
- Fixed modularity calculation. Igraph was converted incorrectly (previously)
- Minor improvements in module / predictor structure
- Changed evaluation code to distinct between overlapping gt and input. Calculate modularity if input it not overlapping
- Fixed evaluation by adding fallback value for community overlap (to avoid modifying existing code)

### Documentation

- Updated readme and requirements regarding usage of kaggle and mamba

### Features

- Added changelog generation
- Added label/cluster entropy metric to detect early on the quality of cluster assignments
- Updated comopt and homophily notebooks to include new metrics and a few small fixes
- Added pretraining + join cluster + embedding optimization using kl div and target distribution reinforcement (adds more confident assignments, empathises more confident points), (still tends to sometimes go to entropy 0)
- Added nn clustering module and contrastive loss
- Added contrastive opt with cluster centers
- Added euclidean variant of centroid based contrastive loss optimization
- Added GraphSAGE based repr learning
- Added GraphSAGE link prediction models
- Split simple clustering flow into modules
- Added homogenous variant of SW dataset with better split into validation and train dataset
- Added node2vec based clustering
- Added node2vec script and community optimization with a better modularity approximation
- Added CE loss to n2v
- Added pos+neg neighbourhood sampling based learning to GraphSAGE based model
- Added notebook for pos+neg neighbourhood sampling (GraphSAGE)
- Added max margin loss and community optimization using mm loss
- Added cluster initializations to graphsage max margin
- Added star wars dataset into snapshot splitting
- Added graphsage based second pass for temporal homophiliy optimization
- Added community optimization using temporal features
- Added standalone tensorflow projector to preview trained embeddings
- Added saving for positional and temporal embeddings separately

### Miscellaneous Tasks

- Updated changelog
- Updated changelog
- Updated readme with info about environment activation and java version

## [2022.06] - 2022-02-08

### Bug Fixes

- Updated louvain to use node renumbering
- Added time order based node relabeling
- Added time order based node relabeling

### Features

- Updated evaluation for cleaner community cleaning and evaluation
- Added GEMSEC sourcecode
- Added neighborhood_size parameter to ANGEL to fix empty communities issues in DBLP and IMDB datasets
- Added dynamo baseline
- Added docker config and run script to dynamo baseline
- Renamed dynamo baseline
- Updated baseline configs
- Added benchmarking results
- Added collected benchmark results
- Added benchmark metric plots
- Added star wars dataset feature engineering
- Added star-wars feature extraction and formatting
- Added link prediction test configuration on the star wars dataset
- Updated demo link prediction for visualization
- Added evaluation metric calculation for kmeans based link prediction tasks
- Added experimentation with link prediction and graph repr frameworks
- Geometric pytorch based prediction using graph convolutions; Added link based graph sampling and loader
- Added pytorch lightning integration for training loop callbacks and data loading multiplexing
- Updated naming of deprecated stellargraph notebooks
- Added implementation of homophily based loss and experiment for clustering based on that
- Added base class for training module for simpler metric logging
- Added cosine based homophily loss calculation
- Updated todo list with goals for objective function experiments

## [2022.04] - 2022-01-26

### Bug Fixes

- Fixed angel baseline to use comlist and edgelist from input paths
- Fixed issue undirected edges are loaded as directed into igraph
- Fixed dual histogram visualization by removing 0th x value when using log scale
- Fixed duplicate and missing nodes in imdb 5000 dataset
- Fixed DBLP datasets by cleaning up node ids
- Fixed community serialization and deserialization
- Fixed eucore dataset to use 0 indexed communities
- Added timeouts to benchmarks added safety checks when no communities are found
- Fixed issue with wrong version selection during benchmarking sweep

### Features

- Added starwars dataset preprocessing
- Updated dataset table
- Added benchmark tuning config
- Added a simple export to edgelist script
- Added a simple baseline running script
- Added helper methods to schema for label and temporal property filtering
- Added loading module for loading and converting between graph types
- Added dataset metric visualization utilities
- Added initial dataset metrics/exploration notebook
- Updated todo list
- Updated visualization to show community based statistics
- Added temporal properties to graph schema
- Added dataset analysis notebooks and added support for temporal edges
- Added dataset analysis notebooks for remaining datasets DBLP-* and sx-*
- Added a script to split datasets into edgelist snapshots
- Added dataset versioning
- Updated benchmark execution config
- Updated greene DCD baseline for benchmark use
- Added a unified community tracking/matching format. Updated ANGEL and GreeneDD baselines
- Added baseline benchmark configurations and updated baseline run scripts
- Added benchmark parameter sweep formatting and merging
- Added ground truth reformatting and registering in schema for relevant datasets
- Added basic setup for evaluation metrics. Added nmi
- Added setup for hyperparameter sweeps
- Added tests for format conversion; Added ordering for comlist format
- Added nf1 score calculation
- Bumped pandoc scripts version
- Added additional metrics modularity and conductance
- Added stricter typing in format functions; Updated tests
- Added graph schema definition
- Added named comlist format
- Added dataset version creation script
- Added ground truth community splitting over snapshots
- Added dataset version configs for a few datasets
- Added basic evaluation structure
- Added portable run scripts for execution and evaluation
- Added baseline hyperparameter tuning and recording in W&B
- Added more evaluation metrics
- Updated dataset version file naming and structure
- Updated benchmark configs and their tags.
- Fixed ESPRA to use 0-indexed edgelist and to output 0-indexed communities
- Added additional dataset version configs
- Configured ANGEL benchmark
- Updated baselines and added benchmarks
- Added dataset overview generation
- Added house of representatives dataset versions and analysis
- Added synthetic dataset tag and exclusion from scripts
- Added support for synthetic benchmark graph generation
- Updated benchmarking config
- Updated benchmark configs; Added visualization export
- Added some simple documentation

### Refactor

- Moved schemas into schemas folder to avoid package resolve conflicts
- Moved preprocessing notebooks into notebook dir
- Added custom format loading and writing
- Updated baselines to use new file format naming convention
- Refactored format and schema namespacing
- Updated dataset and datagraph schemas
- Moved datasets files into config folder
- Updated standard formats to be 0-indexed
- Moved all dynamic data to storage folder
- Updated gitignore

## [2022.02] - 2022-01-12

### Bug Fixes

- Fixed issue where User label is appended to every node label.

### Features

- Added conda environment and environment locking scripts
- Added baselines which have their source code public
- Added baselines which have their source code public
- Bumped pandoc scripts version
- Added connections and logging configs
- Moved docker-compose and updated env for database based scripts
- Added logging and constants shared utils
- Added config shared utils (connection config)
- Added parquet to neo4j schema defintion and building
- Added basic neo4j to gephi export demo script
- Updated requirements and removed views (they are outdated)
- Added export to neo4j script
- Added enron email dataset preprocessing and updated zachary
- Added support for schema merging
- Added automated raw dataset downloading
- Added datasets readme listing datasets and their details
- Added social-distancing-student preprocessing
- Added preprocessing of com-youtube dataset
- Added email-Eu-core dataset preprocessing
- Added misc-football dataset preprocessing
- Added sx-mathoverflow and sx-superuser dataset preprocessing
- Preprocessed imdb-5000-movie-dataset dataset
- Added DBLP dataset preprocessing
- Added DBLP-HCN dataset preprocessing
- Added checking whether database already exists before export to neo4j
- Added activate script to add all relevant paths to Python path
- Added automated export to neo4j
- Added export to graphml script
- Added additional Dynamic Community Detection baselines
- Added ANGEL and ARCHANGEL baselines running script
- Added dockerfile for GreeneDCD baseline
- Added working run script for GreeneDCD baseline
- Added marvel universe dataset to download list
- Added running script for ESPRA baseline
- Added dockerfile for CommunityGAN
- Added run script and docker container for CommunityGAN baseline
- Standardized ANGEL and ESPRA baselines output formats
- Added dockerfile and run configuration for ComE baseline
- Added GEMSEC and LabelPropagation baseline run scripts and docker configs
- Added RESOURCES readme to avoid dumping interesting libraries into the todo list
- Added evolve GCN as dynamic network embedding learning baseline

### Miscellaneous Tasks

- Updated baselines readme
- Updated env and env activation script; Updated todos

### Refactor

- Restructured datasets subproject

### Report

- Added thesis proposal feedback and updated report a bit

