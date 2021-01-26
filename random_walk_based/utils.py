from graph import *
import random
import matplotlib.pyplot as plt

def load_adjacencylist(input_file, undirected=False, chunksize=10000) :
    adjlist = []
    total = 0
    with open(input_file, "r") as f :
        lines = f.readlines()
        for line in lines :
            adjlist.append(list(map(int, line.strip().split(" "))))
    return adjlist

def get_graph(adjlist, name) :
    G = Graph(name)
    for row in adjlist :
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors
    return G

def build_deepwalk_corpus(G, num_paths, path_length, alpha, rand) :
    walks = []
    nodes = list(G.nodes())

    for cnt in range(num_paths) :
        rand.shuffle(nodes)
        for node in nodes :
            walk = G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
            walks.append(walk)
    return walks

def visualize_latent_feature(output_file) :

    color = ["mediumpurple", "lightgreen", "indianred", "mediumturquoise"]
    label_dict = {}
    with open("./results/karate.classification", "r") as label_file :
        lines = label_file.readlines()
        for line in lines :
            line = list(map(int, line.strip().split(" ")))
            label = line[0]
            node_list = line[1:]
            for node in node_list :
                label_dict[node] = label
    
    with open(output_file, "r") as latent_feature_file :
        lines = latent_feature_file.readlines()
        x_coords = []
        y_coords = []
        colors = []
        for line in lines[1:] :
            i, x, y = list(map(float, line.strip().split(" ")))
            x_coords.append(x)
            y_coords.append(y)
            colors.append(color[label_dict[int(i)]])

    plt.scatter(x_coords, y_coords, c=colors)
    plt.title("Representations of Karate Graph")
    plt.savefig("./results/representations.png")    