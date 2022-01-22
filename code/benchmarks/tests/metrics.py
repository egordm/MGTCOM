import unittest

from sklearn.metrics import normalized_mutual_info_score

from datasets.formats import read_comlist
from shared.constants import BENCHMARKS_PATH


# def labellist_to_labels()

class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data_dir = BENCHMARKS_PATH.joinpath('tests/data')
        self.labels_x = read_comlist(self.data_dir.joinpath('predict.labellist'))['cid'].values
        self.labels_y = read_comlist(self.data_dir.joinpath('ground_truth.labellist'))['cid'].values

    def test_nmi(self):
        score = normalized_mutual_info_score(self.labels_x, self.labels_y)
        self.assertAlmostEqual(score, 0.995395850920710, places=8)  # Reference score from ESPRA


if __name__ == '__main__':
    unittest.main()
