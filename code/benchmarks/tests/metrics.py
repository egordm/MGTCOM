import unittest

from benchmarks.metrics import nmi, nf1, modularity, conductance
from datasets.formats import read_comlist, comlist_to_coms, read_edgelist_graph
from shared.constants import BENCHMARKS_PATH


# def labellist_to_labels()

class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data_dir = BENCHMARKS_PATH.joinpath('tests/data')
        self.labels_x = read_comlist(self.data_dir.joinpath('predict.comlist'))
        self.labels_y = read_comlist(self.data_dir.joinpath('ground_truth.comlist'))
        self.G = read_edgelist_graph(self.data_dir.joinpath('testnet.edgelist'), directed=False)

    def test_nmi(self):
        score = nmi(self.labels_x, self.labels_y)
        self.assertAlmostEqual(score, 0.995395850920710, places=8)  # Reference score from ESPRA

    def test_nf1(self):
        coms_x = comlist_to_coms(self.labels_x)
        coms_y = comlist_to_coms(self.labels_y)

        score = nf1(coms_x, coms_y)
        self.assertAlmostEqual(score, 0.96030, places=4)

    def test_modularity(self):
        score = modularity(self.G, self.labels_x)
        # Verified with gephi
        self.assertAlmostEqual(score, 0.7666298582526153, places=5)

    def test_conductance(self):
        coms_x = comlist_to_coms(self.labels_x)
        score = conductance(self.G, coms_x)
        self.assertAlmostEqual(score, 0.19958352550112662, places=5)




if __name__ == '__main__':
    unittest.main()
