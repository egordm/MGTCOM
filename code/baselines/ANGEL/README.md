# ANGEL
[![Test and Coverage (Ubuntu)](https://github.com/GiulioRossetti/ANGEL/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/GiulioRossetti/ANGEL/actions/workflows/test_ubuntu.yml)
[![codecov](https://codecov.io/gh/GiulioRossetti/ANGEL/branch/master/graph/badge.svg?token=3YJOEVK02B)](https://codecov.io/gh/GiulioRossetti/ANGEL)
[![PyPI download month](https://img.shields.io/pypi/dm/angel-cd.svg?color=blue&style=plastic)](https://pypi.python.org/pypi/angel-cd/)

Community discovery in complex networks is an interesting problem with a number of applications, especially in the knowledge extraction task in social and information networks. 
However, many large networks often lack a particular community organization at a global level. 
In these cases, traditional graph partitioning algorithms fail to let the latent knowledge embedded in modular structure emerge, because they impose a top-down global view of a network. 
We propose here a simple local-first approach to community discovery, namely **Angel**, able to unveil the modular organization of real complex networks. 
This is achieved by democratically letting each node vote for the communities it sees surrounding it in its limited view of the global system, i.e. its ego neighborhood, using a label propagation algorithm; finally, the local communities are merged into a global collection. 

Moreover, we provide also an evolution of Angel, namely **ArchAngel**, designed to extract community from evolving network topologies.

**Note:** Angel has been integrated within [CDlib](http://cdlib.readthedocs.io) a python package dedicated to community detection algorithms, check it out!


## Installation
You can easily install the updated version of Angel (and Archangel) by using pip:

```bash
pip install angel-cd
```

or using conda

```bash
conda install -c giuliorossetti angel-cd
```

## Implementation details

*Required input format(s)* 

Angel:
.ncol edgelist (nodes represented with integer ids).

```
node_id0    node_id1
```

ArchAngel:
Extended .ncol edgelist (nodes represented with integer ids).

```
node_id0    node_id1	snapshot_id
```

# Execution
Angel is written in python and requires the following package to run:
- python 3.x
- python-igraph
- networkx
- tqdm

## Angel

```python
import angel as a
an = a.Angel(filename, threshold=0.4, min_comsize=3, outfile_name="angel_communities.txt")
an.execute()
```

Where:
* filename: edgelist filename
* threshold: merging threshold in [0,1]
* min_com_size: minimum size for communities
* out_filename: desired filename for the output 

or alternatively

```python
import angel as a
an = a.Angel(graph=g, threshold=0.4, min_com_size=3, out_filename="communities.txt")
an.execute()
```

Where:
* g: an igraph.Graph object

## ArchAngel

```python
import angel as a
aa = a.ArchAngel(filename, threshold=0.4, match_threshold=0.4, min_com_size=3, outfile_path="./")
aa.execute()
```

Where:
* filename: edgelist filename
* threshold: merging threshold in [0,1]
* match_threshold: cross-time community matching threshold in [0, 1]
* min_com_size: minimum size for communities
* outfile_path: path for algorithm output files 
