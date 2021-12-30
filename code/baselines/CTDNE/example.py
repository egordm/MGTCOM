import numpy as np
import networkx as nx
from ctdne import CTDNE

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)
m = len(graph.edges())
edge2time = {edge: time for edge,time in zip(graph.edges(),(m*np.random.rand(m)).astype(int))}
nx.set_edge_attributes(graph,edge2time,'time')

# Precompute probabilities and generate walks
CTDNE_model = CTDNE(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Embed
model = CTDNE_model.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)