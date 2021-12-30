import random
import igraph
from future.utils import iteritems
import numpy as np
import sys
import time
import tqdm
from collections import Counter

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

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if args[0].verbose:
            sys.stdout.write('Time:  %r %2.2f sec\n' % (method.__name__.strip("_"), te - ts))
            sys.stdout.write('------------------------------------\n')
            sys.stdout.flush()

        return result

    return timed


class Angel(object):
    """
    Angel: Advanced Network Groups Estimate and Localization (igraph implementation)
    """

    def __init__(self, network_filename=None, graph=None, threshold=0.25, min_comsize=3, save=True, outfile_name="angels_coms.txt",
                 dyn=None, verbose=True):
        """
        Constructor

        :param graph: an igraph.Graph object
        :param network_filename: the .ncol network file
        :param threshold: the tolerance required in order to merge communities
        :param min_comsize: minimum desired community size
        :param outfile_name: desired output file name
        :param save: (True|False) whether output the result on file or not
        """

        self.verbose = verbose

        if dyn is None:
            if graph is None:
                self.__read_graph(network_filename)
            else:
                self.G = graph
        else:
            self.G = dyn

        self.threshold = threshold

        if self.threshold < 1:
            self.min_community_size = max([3, min_comsize, int(1. / (1 - self.threshold))])
        else:
            self.min_community_size = min_comsize

        if self.verbose:
            sys.stdout.write("Min. community size: %s\n" % self.min_community_size)
            sys.stdout.flush()

        self.save = save
        self.outfile_name = outfile_name
        self.total_nodes = self.G.vcount()
        self.actual_id = self.total_nodes + 1

        # Additional data structures
        self.all_communities = {}
        self.node2com = {}

    @timeit
    def __read_graph(self, network_filename):
        """
        Read .ncol network file

        :param network_filename: complete path for the .ncol file
        :return: an undirected igraph network
        """
        self.G = igraph.Graph.Read_Ncol(network_filename, directed=False)

    @timeit
    def execute(self):
        """
        Execute Angel algorithm

        :return: a dictionary <node, community_list>
        """

        for ego in tqdm.tqdm(range(0, self.total_nodes), ncols=35, bar_format='Exec: {l_bar}{bar}', disable=not self.verbose):

            # ego_minus_ego node set
            ego_minus_ego_nodes = set(self.G.neighborhood(ego, 1, mode="ALL")) - {ego}
            if len(ego_minus_ego_nodes) >= self.min_community_size:
                community_to_nodes = self.__overlapping_label_propagation(ego_minus_ego_nodes)

                # merging phase
                self.__tpr_merge(community_to_nodes)

        # Clean communities (second-phase merge)
        old_csize = len(self.all_communities)
        diff = old_csize
        it = 1
        while diff > 0:
            self.__clean_communities(it)
            actual_csize = len(self.all_communities)
            diff = old_csize - actual_csize
            old_csize = actual_csize
            it += 1

        if self.verbose:
            sys.stdout.write("\n")

        # Node Lookup
        cms = {}
        idc = 0
        for c in self.all_communities.values():
            ls = [self.G.vs[x]['name'] for x in c]
            if len(ls) >= self.min_community_size:
                cms[idc] = sorted(ls)
                idc += 1

        # output communities
        if self.save and len(cms) > 0:
            out_file_com = open(self.outfile_name, "w")

            for cid, c in iteritems(cms):
                out_file_com.write("%d\t%s\n" % (cid, str(c)))

            out_file_com.flush()
            out_file_com.close()

        return cms

    def __clean_communities(self, it):
        """
        Perform a second merging step to ensure full containment
        """
        sys.stdout.write("")
        sys.stdout.flush()

        com_sorted = sorted(self.all_communities, key=lambda k: len(self.all_communities[k]))

        for c in tqdm.tqdm(com_sorted, ncols=35, bar_format='Clean %s: {l_bar}{bar}' % it, disable=not self.verbose):
            self.__tpr_merge({c: self.all_communities[c]}, clean=True)

    def __tpr_merge(self, community_to_nodes, clean=False):
        """
        Apply F1-merge to ego-network based micro communities

        :param community_to_nodes: dictionary <community_id, node_list>
        :param clean: (True|False) whether the method is called for the cleaning stage or not
        """

        # cycle over micro communities
        for c in community_to_nodes:
            actual_community = community_to_nodes[c]

            # check size constraint
            if len(actual_community) < self.min_community_size:
                return

            # compute community label frequencies
            comsids = []
            for node in actual_community:
                node = self.G.vs[node]
                if node.index in self.node2com:
                    comsids.extend(list(set(self.node2com[node.index])))

            flag = False
            if len(comsids) > 0:
                # identify most frequent community labels
                maxid_set = self.__tpr_filter(len(actual_community), comsids, c)

                if len(maxid_set) > 0:

                    # cycle over merging candidates
                    for current_maxid in maxid_set:

                        # adjust community membership
                        for nid in actual_community:
                            nid = self.G.vs[nid].index

                            if nid in self.node2com:
                                s = self.node2com[nid]
                                if c in s:
                                    del self.node2com[nid][c]
                                if current_maxid not in s:
                                    self.node2com[nid][current_maxid] = None
                            else:
                                self.node2com[nid] = {current_maxid: None}

                        maxid_com = self.all_communities[current_maxid]
                        nc = list((set(maxid_com) | set(actual_community)))
                        self.all_communities[current_maxid] = nc

                        if clean and c in self.all_communities:
                            del self.all_communities[c]
                        flag = True

            # no matching available (and not in the cleaning stage), add the current community
            if not flag and not clean:
                self.all_communities[c] = actual_community
                for node in actual_community:
                    node = self.G.vs[node]
                    if node.index in self.node2com:
                        self.node2com[node.index][c] = None
                    else:
                        self.node2com[node.index] = {c: None}

    def __f1_merge(self, community_to_nodes, clean=False):
        """
        Apply F1-merge to ego-network based micro communities

        :param community_to_nodes: dictionary <community_id, node_list>
        :param clean: (True|False) whether the method is called for the cleaning stage or not
        """

        # cycle over micro communities
        for c in community_to_nodes:
            actual_community = community_to_nodes[c]

            # check size constraint
            if len(actual_community) < self.min_community_size:
                return

            # compute community label frequencies
            comsids = []
            for node in actual_community:
                node = self.G.vs[node]
                if node.index in self.node2com:
                    comsids.extend(self.node2com[node.index])

            flag = False
            if len(comsids) > 0:
                # identify most frequent community labels
                maxid_set = list(self.__tpr_filter(len(actual_community), comsids, c))

                if len(maxid_set) > 0:

                    # cycle over merging candidates
                    for current_maxid in maxid_set:

                        # adjust community membership
                        for nid in actual_community:
                            nid = self.G.vs[nid].index

                            if nid in self.node2com:
                                s = set(self.node2com[nid])
                                if c in s:
                                    s = s - {c}
                                if current_maxid not in s:
                                    s.add(current_maxid)
                                self.node2com[nid] = list(set(s))
                            else:
                                self.node2com[nid] = [current_maxid]

                        maxid_com = self.all_communities[current_maxid]
                        nc = list(set(set(maxid_com) | set(actual_community)))
                        self.all_communities[current_maxid] = nc

                        if clean and c in self.all_communities:
                            del self.all_communities[c]
                        flag = True

            # no matching available (and not in the cleaning stage), add the current community
            if not flag and not clean:
                self.all_communities[c] = actual_community
                for node in actual_community:
                    node = self.G.vs[node]
                    if node.index in self.node2com:
                        self.node2com[node.index].append(c)
                    else:
                        self.node2com[node.index] = [c]

    def __tpr_filter(self, actual_com_size, lst, cid):
        """
        Compute the most common value(s) in a list

        :param lst: the list
        :param cid: community to esclude
        :return: a list containing the most common value(s)
        """
        data = Counter(lst)

        idxs = {k: None for k in data if float(data[k]) / actual_com_size > self.threshold}
        if cid in idxs:
            del idxs[cid]

        return idxs.keys()

    def __overlapping_label_propagation(self, ego_minus_ego_nodes):
        """
        Perform overlapping label propagation on a given ego-minus-ego graph

        :param ego_minus_ego_nodes: ego-minunus-ego node list
        :return: a dictionary <community_id, node_list>
        """

        t = 0
        skip = {}
        node2label = {}
        old_node_to_coms = {}
        ego_minus_ego_nodes = set(ego_minus_ego_nodes)
        total_nodes = len(ego_minus_ego_nodes)

        # set an upper bound to the label propagation iterations
        max_iteration = min(7, np.log2(total_nodes) + 1)

        while t < max_iteration:

            node_to_coms = {}
            count = -total_nodes

            for n in ego_minus_ego_nodes:
                label_freq = {}
                n_neighbors = set(self.G.neighbors(n)) & ego_minus_ego_nodes

                # all nodes have stable community assignment
                if len(skip) == total_nodes:
                    break

                # check if n has a stable community assignment
                if n in skip:
                    continue

                if count == 0:
                    t += 1

                for nn in n_neighbors:
                    communities_nn = [nn]
                    if nn in old_node_to_coms:
                        communities_nn = old_node_to_coms[nn]
                    for nn_c in communities_nn:
                        if nn_c in label_freq:
                            v = label_freq.get(nn_c)
                            label_freq[nn_c] = v + 1
                        else:
                            label_freq[nn_c] = 1

                # first run, random choosing of the communities among the neighbors labels
                if t == 1:
                    if not len(n_neighbors) == 0:
                        r_label = random.sample(label_freq.keys(), 1)
                        node2label[n] = r_label
                        old_node_to_coms[n] = r_label
                    count += 1
                    continue

                # choose the majority
                else:
                    labels = []
                    max_freq = -1

                    for l, c in iteritems(label_freq):
                        if c > max_freq:
                            max_freq = c
                            labels = [l]
                        elif c == max_freq:
                            labels.append(l)

                    node_to_coms[n] = labels
                    old_node_to_coms[n] = node_to_coms[n]
                    node2label[n] = labels

                    # all n's neighbors belong to the same community: stable assignment
                    if max_freq == len(n_neighbors):
                        skip[n] = True
            t += 1

        # build the communities (not reintroducing the ego)
        community_to_nodes = {}
        cidmap = {}
        for n in node2label:
            for l in node2label[n]:

                # get the unique cid
                if l in cidmap:
                    l = cidmap[l]
                else:
                    cidmap[l] = self.gen_nextid()
                    l = cidmap[l]

                if l in community_to_nodes:
                    community_to_nodes[l].append(n)
                else:
                    community_to_nodes[l] = [n]

        return community_to_nodes

    def gen_nextid(self):
        """
        Generate the next unique community id

        :return: new community id
        """
        self.actual_id += 1
        return self.actual_id


if __name__ == "__main__":
    import argparse

    print("------------------------------------")
    print("              iAngel                ")
    print("------------------------------------")
    print("Author: ", __author__)
    print("Email:  ", __contact__)
    print("WWW:    ", __website__)
    print("------------------------------------")

    parser = argparse.ArgumentParser()

    parser.add_argument('network_file', type=str, help='network file (edge list format)')
    parser.add_argument('threshold', type=float, help='merging threshold')
    parser.add_argument('-c', '--min_com_size', type=int, help='minimum community size', default=3)
    parser.add_argument('-s', '--save', type=bool, help='save on file', default=True)
    parser.add_argument('-o', '--out_file', type=str, help='output file', default="angels_coms.txt")

    args = parser.parse_args()
    an = Angel(args.network_file, graph=None, threshold=args.threshold,
               min_comsize=args.min_com_size, save=args.save, outfile_name=args.out_file)

    an.execute()
