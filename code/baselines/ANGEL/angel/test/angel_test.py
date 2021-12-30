import unittest
import angel as a
import os
import glob
import networkx as nx
import igraph as ig


class DemonTestCase(unittest.TestCase):

    def test_angel(self):
        g = nx.karate_club_graph()
        nx.write_edgelist(g, "test.csv", delimiter=" ")

        an = a.Angel("test.csv", threshold=0.6, min_comsize=3, save=True)
        coms = an.execute()
        self.assertEqual(len(coms), 3)

        os.remove("test.csv")
        os.remove("angels_coms.txt")

    def test_angel_OBJ(self):
        G = nx.karate_club_graph()
        g = ig.Graph.TupleList(G.edges(), directed=False)
        g.vs['club'] = list(nx.get_node_attributes(G, 'club').values())

        an = a.Angel(graph=g, threshold=0.6, min_comsize=3, save=True)
        coms = an.execute()
        self.assertEqual(len(coms), 3)

        os.remove("angels_coms.txt")

    def test_archangel(self):

        aa = a.ArchAngel("%s/sgraph.txt" % os.path.dirname(os.path.abspath(__file__)),
                         threshold=0.4, match_threshold=0.3)
        coms = aa.execute()
        self.assertEqual(len(coms), 6)

        for f in glob.glob("ArchAngel*"):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
