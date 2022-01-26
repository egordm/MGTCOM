# Dynamic Community Detection Workspace

## Structure
* baselines/ - Code for baseline algorithms
  * baselines/ - Code for baseline algorithms
    * scripts/ - Scripts for baseline running, evaluation and tuning
* benchmarks/ - Benchmarking and evaluation code
* configs/
  * benchmarks/ - Benchmark running and tuning configurations
  * datasets/ - Dataset and their version configurations 
  * connections.yml - Configuration file for external services W&B and Neo4J
* datasets/
  * datasets/ - Dataset preprocessing, analysis and dataset versioning code
    * scripts/ - Scripts for dataset preprocessing, dataset versioning and visualzation
  * notebooks/
    * preprocessing/ - Dataset preprocessing notebooks
    * analysis/ - Dataset analysis notebooks
* shared/ - Shared code for all projects (e.g. graph schemas, config schema)
* storage/ - Storage folder for datasets and results
