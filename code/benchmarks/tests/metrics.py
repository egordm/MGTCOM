import unittest

from benchmarks.benchmarks.metrics import nmi, nf1
from datasets.formats import read_comlist, comlist_to_coms
from shared.constants import BENCHMARKS_PATH


# def labellist_to_labels()

class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data_dir = BENCHMARKS_PATH.joinpath('tests/data')
        self.labels_x = read_comlist(self.data_dir.joinpath('predict.comlist'))
        self.labels_y = read_comlist(self.data_dir.joinpath('ground_truth.comlist'))

    def test_nmi(self):
        score = nmi(self.labels_x, self.labels_y)
        self.assertAlmostEqual(score, 0.995395850920710, places=8)  # Reference score from ESPRA

    def test_nf1(self):
        coms_x = comlist_to_coms(self.labels_x)
        coms_y = comlist_to_coms(self.labels_y)

        score = nf1(coms_x, coms_y)
        self.assertAlmostEqual(score, 0.96030, places=4)


if __name__ == '__main__':
    unittest.main()
