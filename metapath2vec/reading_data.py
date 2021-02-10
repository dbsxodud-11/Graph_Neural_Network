import numpy as np
import torch
from torch.utils.data import Dataset

class DataReader :

    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, input_path, min_count, care_type) :
        self.negatives = []
        self.discards = []
        self.negpos = []
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = in
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()