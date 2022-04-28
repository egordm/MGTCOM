from typing import Dict

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch import Tensor


def community_gt_metrics(z: Tensor, gt: Tensor) -> Dict[str, float]:
    return {
        'nmi': normalized_mutual_info_score(gt, z),
        'ari': adjusted_rand_score(gt, z),
    }
