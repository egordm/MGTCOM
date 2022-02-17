import json
import random
import string

import pandas as pd
import torch

from shared.constants import BASE_PATH

PROJECTOR_PATH = BASE_PATH.joinpath('experiments', 'embedding-projector')
PROJECTOR_DATA_PATH = PROJECTOR_PATH.joinpath('oss_data')


def save_projector(
        name: str,
        embeddings: torch.Tensor,
        metadata: pd.DataFrame,
):
    file_alias = name.lower().replace(' ', '_') # + '-' + ''.join(random.choices(string.ascii_letters,k=6))
    embeddings_file = PROJECTOR_DATA_PATH.joinpath(f'{file_alias}_tensors.bytes')
    metadata_file = PROJECTOR_DATA_PATH.joinpath(f'{file_alias}_metadata.tsv')
    config_file = PROJECTOR_DATA_PATH.joinpath('oss_demo_projector_config.json')

    embeddings.numpy().tofile(str(embeddings_file))
    metadata.to_csv(str(metadata_file), index=False, sep='\t', header=True)

    with config_file.open('r') as f:
        config = json.load(f)

    config['embeddings'] = [x for x in config['embeddings'] if x['tensorName'] != name]
    config['embeddings'].append({
        'tensorName': name,
        'tensorShape': list(embeddings.shape),
        'tensorPath': str(embeddings_file.relative_to(PROJECTOR_PATH)),
        'metadataPath': str(metadata_file.relative_to(PROJECTOR_PATH)),
    })

    with config_file.open('w') as f:
        json.dump(config, f)