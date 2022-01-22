import unittest

from datasets.formats import read_comlist, comlist_to_coms, coms_to_comlist
from shared.constants import BENCHMARKS_PATH


class TestFormats(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data_dir = BENCHMARKS_PATH.joinpath('tests/data')
        self.labels_x = read_comlist(self.data_dir.joinpath('predict.comlist'))
        self.labels_y = read_comlist(self.data_dir.joinpath('ground_truth.comlist'))

    def test_convert_comlist_coms(self):
        coms = comlist_to_coms(self.labels_x)
        comlist = coms_to_comlist(coms)
        self.assertTrue((self.labels_x == comlist).all().all())


if __name__ == '__main__':
    unittest.main()
