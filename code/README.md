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


## Environment Setup
* Make sure you have `conda` installed
  * If not, follow the [offical guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it
* To speedup installation process you can also install `mamba`
  * Follow these steps to install it: [mamba](https://github.com/mamba-org/mamba)
    * tldr: `conda install mamba -n base -c conda-forge`
  * This is not necessary as mamba and conda commands are interchangeable
* Create your conda environment:
  * From the dev config: `mamba env create -f environment.yml`
  * From the lock file: `mamba env create -f environment.lock.yml`
* Some parts of the project use `pyspark`
  * Ensure you have the right java version installed using `java -version`
  * Project is known to work with: openjdk11


## Working with the code
#### Activating the environment
Make sure you have conda installed beforehand.

```bash
source ./activate.sh
```