import argparse
import numpy as np
import networkx as nx

from gensim.models import Word2Vec
from graph import Node2Vec_Graph
from utils import *

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="../example_graphs/karate.adjlist",
                      help='Input graph file')
    parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')
    parser.add_argument('--output', default="./results/karate_deepwalk.embeddings",
                        help='Output representation file')
    parser.add_argument('--representation-size', default=2, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')
    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')
    parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')
    parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')
    parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)
    args = parser.parse_args()
    return args

def read_graph() :
    G = nx.read_adjlist(args.input, nodetype=int)
    return G

def learn_embeddings(walks) :
    
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size)
    model.wv.save_word2vec_format(args.output)

def process(args) :

    nx_G = read_graph()
    G = Node2Vec_Graph(nx_G, args.undirected, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.number_walks, args.walk_length)
    learn_embeddings(walks)
    visualize_latent_feature(args.output, "node2vec")

if __name__ == "__main__" :
    args = parse_args()
    process(args)