from collections import defaultdict
import numpy as np
import networkx as nx
import random

class Graph(defaultdict):
    def __init__(self, name):
        super(Graph, self).__init__(list)
        self.name = name
    
    def nodes(self):
        return self.keys()
    
    def subgraph(self, nodes={}):
        subgraph = Graph()
        for node in nodes :
            if node in self.keys():
                subgraph[node] = list(filter(lambda x: x in nodes, self[node]))
    
    def make_undirected(self):
        for v in list(self.keys()) :
            for n_v in self[v] :
                if v != n_v :
                    self[n_v].append(v) # two-way connection

    def random_walk(self, path_length, alpha, rand, start) :

        path = [start]
        for i in range(path_length-1) :
            cur_node = path[-1]
            if len(self[cur_node]) > 0 :
                if rand.random() >= alpha :
                    path.append(rand.choice(self[cur_node]))
                else :
                    path.append(path[0]) # restart from start node
            else :
                break # no vertices to go
        return list(map(str, path))

class Node2Vec_Graph:

    def __init__(self, nx_G, is_undirected, p, q) :
        self.G = nx_G
        self.is_undirected = is_undirected
        self.p = p
        self.q = q

    def preprocess_transition_probs(self) :

        alias_nodes = {}
        for node in self.G.nodes() :
            unnormalized_probs = [1 for _ in self.G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = list(map(lambda x : float(x)/norm_const, unnormalized_probs))
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}
        if self.is_undirected :
            for edge in self.G.edges() :
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def get_alias_edge(self, src, dst) :

        unnormalized_probs = []
        for dst_nbr in sorted(self.G.neighbors(dst)):
            if dst_nbr == src : # d = 0
                unnormalized_probs.append(1/self.p)
            elif self.G.has_edge(dst_nbr, src) : # d = 1
                unnormalized_probs.append(1)
            else : # d = 2
                unnormalized_probs.append(1/self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = list(map(lambda x: float(x)/norm_const, unnormalized_probs))    
        return self.alias_setup(normalized_probs)

    def alias_setup(self, probs) :
        # Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/ for details
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q

    def alias_draw(self, J, q):
        # Draw sample from a non-uniform discrete distribution using alias sampling.
        K = len(J)

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def simulate_walks(self, num_walks, walk_length) :
        walks = []
        nodes = list(self.G.nodes())
        for walk_iter in range(num_walks) :
            random.shuffle(nodes)
            for node in nodes :
                walks.append(self.node2vec_walk(walk_length, start=node))
        return walks

    def node2vec_walk(self, walk_length, start) :
        walk = [start]
        for i in range(walk_length-1) :
            cur = walk[-1]
            cur_nbrs = list(sorted(self.G.neighbors(cur)))
            if len(cur_nbrs) > 0 :
                if len(walk) == 1 :
                    walk.append(cur_nbrs[self.alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else :
                    prev = walk[-2]
                    walk.append(cur_nbrs[self.alias_draw(self.alias_edges[(prev, cur)][0], self.alias_edges[(prev, cur)][1])])    
            else :
                break
        return list(map(str, walk))