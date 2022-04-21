from datasets import StarWars
from ml.evaluation import extract_edge_prediction_pairs

dataset = StarWars()
data = dataset.data

epochs = 200

pairs, labels = extract_edge_prediction_pairs(
    data.edge_index, data.num_nodes, getattr(data, f'edge_test_mask'),
    max_samples=5000
)