# Datasets
## Usage
#### Download all the datasets in one go:
```bash
make download
```

> Note: Some datasets download from kaggle. To download these datasets you first need to [set up kaggle cli](https://github.com/Kaggle/kaggle-api#api-credentials).

#### Preprocess the datasets by running notebooks in: 
`notebooks/preprocessing`

You can do this by starting jupyter notebook in the root directory using: `jupyter notebook`

> Note: Check root readme on how to activate the virtual environment.

> Note: Some parts of the project use `pyspark`
> 
> Ensure you have the right java version installed using `java -version`
> 
> Project is known to work with: openjdk11

#### Export all the datasets to neo4j
```bash
make export_neo4j
```

#### Export all the datasets to graphml
```bash
make export_graphml
```

#### Create versions for all the datasets
```bash
make version
```


## Used Datasets

| Dataset                                                                                                | Reference                                                                 | Nodes   | Edges   | Node/Edge Types | Properties |
|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|---------|---------|-----------------|------------|
| [Zachary karate club](http://konect.cc/networks/ucidata-zachary/)                                      | [@zacharyInformationFlowModel1976]                                        | 34      | 78      | 1/1             | S/Y        |
| [Football](https://networkrepository.com/misc-football.php)                                            | [@girvanCommunityStructureSocial2002]                                     | 115     | 613     | 1/1             | S/N        |
| [Star Wars Social](https://www.kaggle.com/ruchi798/star-wars)                                          | [@StarWarsSocial]                                                         | 113     | 1599    | 1/2             | T/N        |
| [Enron](https://www.kaggle.com/wcukierski/enron-email-dataset/)                                        | [@vanburenEnronDatasetResearch2009]                                       | 605076  | 4179878 | 2/3             | T/Y        |
| [IMDB 5000](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)                               | [@IMDB5000Movie]                                                          | 16615   | 52251   | 4/4             | T/N        |
| [DBLP-HCN](https://data.mendeley.com/datasets/t4xmpbrr6v/1)                                            | [@tangArnetMinerExtractionMining2008; @yangDefiningEvaluatingNetwork2012] | 10687   | 16533   | 3/2             | T/N        |
| [DBLP-V1](https://www.aminer.org/citation)                                                             | [@tangArnetMinerExtractionMining2008; @yangDefiningEvaluatingNetwork2012] | 1238145 | 2475740 | 3/3             | T/N        |
| [DBLP-V3](https://www.aminer.org/citation)                                                             | [@tangArnetMinerExtractionMining2008; @yangDefiningEvaluatingNetwork2012] | 2677098 | 8225508 | 3/3             | T/N        |
| [sx-mathoverflow](https://snap.stanford.edu/data/sx-mathoverflow.html)                                 | [@SNAPNetworkDatasets]                                                    | 24818   | 506550  | 1/3             | T/N        |
| [sx-superuser](https://snap.stanford.edu/data/sx-superuser.html)                                       | [@SNAPNetworkDatasets]                                                    | 194085  | 1443339 | 1/3             | T/N        |
| [Eu-core network](https://snap.stanford.edu/data/email-Eu-core.html)                                   | [@SNAPNetworkDatasets]                                                    | 1005    | 25571   | 1/1             | S/Y        |
| [com-Youtube](https://snap.stanford.edu/data/com-Youtube.html)                                         | [@SNAPNetworkDatasets]                                                    | 1134891 | 298762  | 1/1             | S/Y        |
| [116th House of Representatives](https://www.kaggle.com/aavigan/house-of-representatives-congress-116) | [@116thHouseRepresentatives]                                              | 1134891 | 298762  | 1/1             | S/Y        |
| [social-distancing-student]()                                                                          | [@wangPublicSentimentGovernmental2020]                                    | 93433   | 3710183 | 3/7             | T/N        |

## Wishlist
- [ ] https://www.kaggle.com/phamvudung/imdb-dataset
- [ ] https://www.kaggle.com/wolfram77/graphs-communities (doenst contain timestamps)
- [ ] https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_interactions.csv
- [ ] https://www.kaggle.com/gpreda/reddit-wallstreetsbets-posts
- [ ] https://www.kaggle.com/gpreda/birds-arent-real
- [ ] https://www.kaggle.com/lakritidis/identifying-influential-bloggers-techcrunch
- [ ] https://www.kaggle.com/andreagarritano/deezer-social-networks
- [ ] https://www.kaggle.com/ellipticco/elliptic-data-set
- [ ] https://www.kaggle.com/hugomathien/soccer
- [ ] https://github.com/GiulioRossetti/cdlib_datasets (the synthetically generated datasets)
- [ ] https://github.com/PanShi2016/Community_Detection/tree/master/Datasets/Real_Data/Original_Data/Flickr]
- [ ] https://networkrepository.com/citeseer.php