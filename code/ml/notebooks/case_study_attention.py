import torch
from tqdm import tqdm

from ml.executors.mgcom_combi_executor import MGCOMCombiExecutor
from ml.utils import dict_mapv, dict_catv, merge_dicts

executor = MGCOMCombiExecutor()
executor.cli('ac_config.yaml', fit=False)
u = 0

emb = []
Att_topo = []
Att_tempo = []

with torch.no_grad():
    model = executor.model
    for batch in tqdm(executor.datamodule.predict_dataloader()):
        Z_emb = model.feat_net.forward(batch)

        emb.append(Z_emb)
        Z_att = dict_mapv(Z_emb, lambda x: torch.sigmoid(model.topo_net(x)))
        Att_topo.append(Z_att)
        Z_att = dict_mapv(Z_emb, lambda x: torch.sigmoid(model.tempo_net(x)))
        Att_tempo.append(Z_att)
        u = 0

Z = merge_dicts(emb, lambda xs: torch.cat(xs, dim=0))
Att_topo = merge_dicts(Att_topo, lambda xs: torch.cat(xs, dim=0))
Att_tempo = merge_dicts(Att_tempo, lambda xs: torch.cat(xs, dim=0))

torch.save({
    'Z': Z,
    'Att_topo': Att_topo,
    'Att_tempo': Att_tempo,
}, 'ac_att.pt')
a = 0