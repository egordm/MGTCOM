<!-- markdownlint-disable -->
<div id="top"></div>
<div align="center">
    <h1>MGTCOM</h1>
    <p>
        <b>Community Detection in Temporal
Multimodal Graphs</br>Official implementation</b>
    </p>
</div>
<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a>
</p>
<!-- markdownlint-enable -->

## Features
MGTCOM Framework is meant to be used as a community detection algorithm in temporal multimodal graphs.
The framework learns temporally and topologically aware embeddings and detects communities in tandem.

### Implemented baselines:

* GraphSAGE[^1]
* Node2Vec[^2]
* ComE[^3]
* GEMSEC[^4]
* CP-GNN[^5]
* CTDNE[^6]

[^1]: W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, Red Hook, NY, USA, Dec. 2017, pp. 1025–1035.
[^2]: A. Grover and J. Leskovec, “node2vec: Scalable Feature Learning for Networks,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, NY, USA, Aug. 2016, pp. 855–864. doi: 10.1145/2939672.2939754.
[^3]: S. Cavallari, V. W. Zheng, H. Cai, K. C.-C. Chang, and E. Cambria, “Learning Community Embedding with Community Detection and Node Embedding on Graphs,” in Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, Singapore Singapore, Nov. 2017, pp. 377–386. doi: 10.1145/3132847.3132925.
[^4]: B. Rozemberczki, R. Davies, R. Sarkar, and C. Sutton, “GEMSEC: graph embedding with self clustering,” in Proceedings of the 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, New York, NY, USA, Aug. 2019, pp. 65–72. doi: 10.1145/3341161.3342890.
[^5]: L. Luo, Y. Fang, X. Cao, X. Zhang, and W. Zhang, “Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model,” in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, New York, NY, USA: Association for Computing Machinery, 2021, pp. 1170–1180. Accessed: Jan. 08, 2022. [Online]. Available: https://doi.org/10.1145/3459637.3482250
[^6]: G. H. Nguyen, J. B. Lee, R. A. Rossi, N. K. Ahmed, E. Koh, and S. Kim, “Continuous-Time Dynamic Network Embeddings,” in Companion of the The Web Conference 2018 on The Web Conference 2018 - WWW ’18, Lyon, France, 2018, pp. 969–976. doi: 10.1145/3184558.3191526.


### Model Variants
* MGTCOM: Learns temporal and topological embeddings
* MGTCOM Topo: Learns topological embeddings
* MGTCOM Tempo: Learns temporal embeddings

### Used Datasets
* DBLP[^7]
* ICEWS[^8]
* IMDB5000[^9]
* SocialDistancingStudents[^10]
* Cora[^11]
* PubMed[^11]

[^7]: J. Yang and J. Leskovec, “Defining and evaluating network communities based on ground-truth,” in Proceedings of the ACM SIGKDD Workshop on Mining Data Semantics, New York, NY, USA, Aug. 2012, pp. 1–8. doi: 10.1145/2350190.2350193.
[^8]: A. García-Durán, S. Dumancic, and M. Niepert, “Learning Sequence Encoders for Temporal Knowledge Graph Completion,” Jan. 2018, pp. 4816–4821. doi: 10.18653/v1/D18-1516.
[^9]: “IMDB 5000 Movie Dataset.” https://kaggle.com/carolzhangdc/imdb-5000-movie-dataset (accessed Jan. 14, 2022).
[^10]: S. Wang, M. Schraagen, E. Tjong Kim Sang, and M. Dastani, “Public Sentiment on Governmental COVID-19 Measures in Dutch Social Media,” presented at the EMNLP-NLP-COVID19 2020, Online, Dec. 2020. doi: 10.18653/v1/2020.nlpcovid19-2.17.
[^11]: Z. Yang, W. W. Cohen, and R. Salakhutdinov, “Revisiting semi-supervised learning with graph embeddings,” in Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, New York, NY, USA, Jun. 2016, pp. 40–48.


## Installation
> **System Requirements:** 
> 
> Note that while conda installs most of the compiled dependencies, the system still needs to have an up-to-date `glibc` version (2.27 and up). 
> 
> Check your version using following command `ldd --version`

* Install either [Anaconda](https://www.anaconda.com/), [Miniconda](https://conda.io/miniconda.html), or [Mamba](https://mamba.readthedocs.io/en/latest/) (recommended).
* Set up the environment:
  * mamba env update --prefix=./env --f environment.yml --prune
  * For cpu only usage replace `pytorch::cudatoolkit=11.3` with `pytorch::cpuonly` in the environment.yml file.
* Activate the environment: `source activate.sh`
* Extract preprocessed datasets:
  * Download the preprocessed datasets from [Drive](https://drive.google.com/file/d/1zNVf4-1_xT84dzTH86kMN0sw7Mj0iV-z/view?usp=sharing)
  * `mkdir -p ./storage/cache/dataset`
  * `cp ./datasets.zip ./storage/cache/dataset/datasets.zip`
  * `cd ./storage/cache/dataset && unzip datasets.zip`

## Examples
The executors are found in `ml/ml/executors` directory.

### Viewing parameters of an axecutor:
```shell
python ml/ml/executors/mgcom_combi_executor.py -h
```

### Running an axecutor:
```shell
python ml/ml/executors/mgcom_combi_executor.py --repr_dim=64
```