import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage

from datasets.transforms.to_homogeneous import to_homogeneous


def homogenify(data: HeteroData):
    hdata = to_homogeneous(data,
        node_attrs=None, edge_attrs=None,
        add_node_type=False, add_edge_type=False
    )
    hdata.node_stores[0]._node_type_names = ['0']
    hdata.node_stores[0]._edge_type_names = [('0', '0', '0')]

    hetdata = hdata.to_heterogeneous()
    for store in hetdata.stores:
        keys = list(hetdata.keys)
        for key in keys:
            if key == 'edge_index':
                continue

            if key == 'x':
                continue

            if key in store:
                if 'edge_' in key and isinstance(store, EdgeStorage):
                    store[key.replace('edge_', '')] = store[key]
                elif 'node_' in key and isinstance(store, NodeStorage):
                    store[key.replace('node_', '')] = store[key]

                del store[key]

    feat_dim = max(x.shape[1] for _, x in data.x_dict.items())
    X = torch.zeros(data.num_nodes, feat_dim)
    offset = 0
    for key in data.node_types:
        x = data[key].x
        X[offset:offset + x.shape[0], :x.shape[1]] = x
        offset += x.shape[0]

    hetdata['0'].x = X

    return hetdata
