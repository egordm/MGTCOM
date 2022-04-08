import unittest

from datasets import StarWars, DBLPHCN, IMDB5000, SocialDistancingStudents, ICEWS0515, HouseOfRepresentativesCongress116


class TestDatasets(unittest.TestCase):
    def test_imdb(self):
        dataset = IMDB5000()

    def test_starwars(self):
        dataset = StarWars()

    def test_dblphcn(self):
        dataset = DBLPHCN()

    def test_students(self):
        dataset = SocialDistancingStudents()

    def test_icews(self):
        dataset = ICEWS0515()

    def test_hor(self):
        dataset = HouseOfRepresentativesCongress116()
