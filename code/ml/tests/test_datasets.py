import unittest

from ml.datasets import IMDB5000, StarWars, DBLPHCN


class TestDatasets(unittest.TestCase):
    def test_imdb(self):
        dataset = IMDB5000()

    def test_starwars(self):
        dataset = StarWars()

    def test_dblphcn(self):
        dataset = DBLPHCN()
