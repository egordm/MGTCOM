from datasets import StarWars, DBLPHCN
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.data.samplers.cpgnn_sampler import CPGNNSampler

dataset = StarWars()
# dataset = DBLPHCN()

hdata = to_homogeneous(dataset.data)

sampler = CPGNNSampler(hdata.edge_index, hdata.num_nodes)

u = 0