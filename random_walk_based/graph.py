from collections import defaultdict

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

    