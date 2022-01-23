import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from shared.schema import DatasetSchema, GraphSchema
from shared.graph import DataGraph, igraph_to_nx

from datasets.visualization import plot_explore_dual_histogram, show_top_k_nodes, show_top_k_stacked_nodes
DATASET = DatasetSchema.load_schema('imdb-5000-movie-dataset')
schema = GraphSchema.from_dataset(DATASET)
G = DataGraph.from_schema(schema)

u = 0