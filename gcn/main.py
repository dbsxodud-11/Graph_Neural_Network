import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    
    args = parser.parse_args()
    return args

def main(args) :
    # Load Data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
    # Model and Optimizer
    model = GCN(n_features=features.shape[1],
                n_hidden=args.hidden,
                n_class=labels.max().item()+1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    print()
    print("Training...")
    for epoch in range(args.epochs) :
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode :
            model.eval()
            output = model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print(f"Epoch : {epoch+1}")
            print(f"Loss : {loss_val.item()}")
            print(f"Accuracy : {acc_val.item()}")
    print()
    print("Testing...")
    # Test Model
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Final Results")
    print(f"Loss : {loss_test.item()}")
    print(f"Accuracy : {acc_test.item()}")

if __name__ == "__main__" :
    args = parse_args()
    main(args)
    