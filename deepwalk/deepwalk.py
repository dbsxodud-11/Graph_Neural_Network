import argparse
from utils import *
from gensim.models import Word2Vec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', default='adjlist',
                      help='File format of input file')
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
    args = parser.parse_args()
    return args

def process(args) :
    name = "karate"
    if args.format == "adjlist" :
        adjlist = load_adjacencylist(args.input, undirected=args.undirected)
        G = get_graph(adjlist, name)
    else :
        raise Exception(f"Unknown file format : {args.format}")
    
    print(f"Target Graph: {G.name}")

    num_walks = len(G.nodes()) * args.number_walks 
    data_size = num_walks * args.walk_length

    print("Walking...")
    walks = build_deepwalk_corpus(G, num_paths=args.number_walks,
                                  path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    print("Traning...")
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size)
    model.wv.save_word2vec_format(args.output)

    print("Visualize Results...")
    visualize_latent_feature(args.output)

if __name__ == "__main__" :
    args = parse_args()
    process(args)