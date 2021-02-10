import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
from reading_data import DataReader, Metapath2vecDataset
from model import SkipGramModel
from download import AminerDataset, CustomDataset


class Metapath2VecTrainer :

    def __init__(self, args) :

        self.data = DataReader(self.input_path, args.min_count, args.care_type)
        









if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="metapath2vec")
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--factors', default=128, type=int)
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--iterations', default=5, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")
    args = parser.parse_args()
    m2v = Metapath2VecTrainer(args)
    m2v.train()