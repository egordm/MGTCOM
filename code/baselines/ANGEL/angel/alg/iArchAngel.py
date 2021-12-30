import sys
import time
from collections import Counter

import igraph
import tqdm
from future.utils import iteritems

from angel.alg import iAngel as an

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"


def timeit(method):
    """
    Decorator: Compute the execution time of a function
    :param method: the function
    :return: the method runtime
    """

    def timed(*arguments, **kw):
        ts = time.time()
        result = method(*arguments, **kw)
        te = time.time()

        sys.stdout.write('Time:  %r %2.2f sec\n' % (method.__name__.strip("_"), te - ts))
        sys.stdout.write('------------------------------------\n')
        sys.stdout.flush()
        return result

    return timed


class ArchAngel(object):

    def __init__(self, network_filename, threshold=0.25, match_threshold=0.25,
                 min_comsize=3, save=True, outfile_path=""):
        """
        Constructor

        :param network_filename: the .ncol network file
        :param threshold: the tolerance required in order to merge communities
        :param min_comsize: minimum desired community size
        :param outfile_name: desired output file name
        :param save: (True|False) whether output the result on file or not
        """

        self.network_filename = network_filename
        self.threshold = threshold
        self.match_threshold = match_threshold

        if self.threshold < 1:
            self.min_community_size = max([3, min_comsize, int(1. / (1 - self.threshold))])
        else:
            self.min_community_size = min_comsize

        self.save = save
        self.outfile_name = "%sArchAngel_coms" % outfile_path
        self.slices_ids = []
        self.snapshot_to_coms = {}

    def __read_snapshot(self, network_filename):
        """
        Read .ncol dynamic network file

        :param network_filename: complete path for the .ncol file
        :return: an undirected igraph network
        """

        previous_slice, actual_slice = None, None
        edge_list = []

        with open(network_filename) as nf:
            ln = 0
            for line in nf:
                edge = line.rstrip().split("\t")
                actual_slice = edge[2]
                if ln == 0:
                    previous_slice = edge[2]
                    ln += 1

                if actual_slice != previous_slice:

                    vertices = set()
                    for ev in edge_list:
                        vertices.update(ev)
                    vertices = sorted(vertices)
                    g = igraph.Graph()
                    g.add_vertices(vertices)
                    g.add_edges(edge_list)
                    edge_list = []

                    self.slices_ids.append(previous_slice)
                    yield g, previous_slice

                    previous_slice = actual_slice

                edge_list.append([edge[0], edge[1]])

            vertices = set()
            for line in edge_list:
                vertices.update(line)
            vertices = sorted(vertices)
            g = igraph.Graph()
            g.add_vertices(vertices)
            g.add_edges(edge_list)
            self.slices_ids.append(actual_slice)
            yield g, actual_slice

    @timeit
    def execute(self):
        """
        Execute ArchAngel algorithm
        """

        for graph, snapshot in tqdm.tqdm(self.__read_snapshot(self.network_filename), ncols=35):
            ag = an.Angel(None, threshold=self.threshold,
                          min_comsize=self.min_community_size,
                          save=self.save, outfile_name="%s_%s.txt" % (self.outfile_name, snapshot),
                          dyn=graph, verbose=False)

            self.snapshot_to_coms[snapshot] = ag.execute()

        with open("%s_ct_matches.csv" % self.outfile_name, "w") as fout:
            fout.write("snapshot_from,snapshot_to,cid_from,cid_to\n")

            # Matching
            for t, fr in enumerate(self.slices_ids[:-1]):
                if t < len(self.slices_ids)-1:
                    mts = self.__tpr_match(fr, self.slices_ids[t+1])

                    # Output cross-time matches
                    for past, future in iteritems(mts):
                        for c_future in future:
                            fout.write("%s,%s,%s,%s\n" % (fr, self.slices_ids[t+1], past, c_future))

        return self.snapshot_to_coms

    def __tpr_match(self, fr, to):
        """
        Apply F1-merge to ego-network based micro communities

        :param community_to_nodes: dictionary <community_id, node_list>
        """

        community_to_nodes_from = self.snapshot_to_coms[fr]
        community_to_nodes_to = self.snapshot_to_coms[to]

        community_events = {}

        if len(community_to_nodes_from) == 0 or len(community_to_nodes_to) == 0:
            return community_events

        # From past to future
        node_to_com_to = {n: cid for cid, nlist in community_to_nodes_to.items() for n in nlist}

        # cycle over micro communities
        for c in community_to_nodes_from:
            actual_community = community_to_nodes_from[c]
            matches = [node_to_com_to[n] for n in actual_community if n in node_to_com_to]
            most_common_coms = {cid: cid_count for cid, cid_count in Counter(matches).most_common()}

            if len(most_common_coms) > 0:
                max_ct = list(most_common_coms.values())[0]

                similarity = float(max_ct)/len(actual_community)
                if similarity >= self.match_threshold:
                    for cf in most_common_coms.keys():
                        if c not in community_events:
                            community_events[c] = [cf]
                        else:
                            community_events[c].append(cf)
                            community_events[c] = list(set(community_events[c]))

        # From future to past
        node_to_com_from = {n: cid for cid, nlist in community_to_nodes_from.items() for n in nlist}

        # cycle over micro communities
        for c in community_to_nodes_to:
            actual_community = community_to_nodes_to[c]
            matches = [node_to_com_from[n] for n in actual_community if n in node_to_com_from]
            most_common_coms = {cid: cid_count for cid, cid_count in Counter(matches).most_common()}

            if len(most_common_coms) > 0:
                max_ct = list(most_common_coms.values())[0]

                similarity = float(max_ct) / len(actual_community)
                if similarity >= self.match_threshold:
                    for cf in most_common_coms:
                        if cf not in community_events:
                            community_events[cf] = [c]
                        else:
                            community_events[cf].append(c)
                            community_events[cf] = list(set(community_events[cf]))

        return community_events


if __name__ == "__main__":
    import argparse

    print("------------------------------------")
    print("             ArchAngel              ")
    print("------------------------------------")
    print("Author: ", __author__)
    print("Email:  ", __contact__)
    print("WWW:    ", __website__)
    print("------------------------------------")

    parser = argparse.ArgumentParser()

    parser.add_argument('network_file', type=str, help='network file (edge list format)')
    parser.add_argument('threshold', type=float, help='merging threshold')
    parser.add_argument('match_threshold', type=float, help='matching threshold')
    parser.add_argument('-c', '--min_com_size', type=int, help='minimum community size', default=3)
    parser.add_argument('-s', '--save', type=bool, help='save on file', default=True)
    parser.add_argument('-o', '--out_file', type=str, help='output file path', default="./")

    args = parser.parse_args()

    aa = ArchAngel(args.network_file, threshold=args.threshold, match_threshold=args.match_threshold,
                   min_comsize=args.min_com_size, save=args.save, outfile_path=args.out_file)
    aa.execute()
