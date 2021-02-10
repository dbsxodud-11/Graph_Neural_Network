import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import style

from utils import *
from models import *


def main() :
    lr = 1e-3
    max_epoch = 300
    n_hidden = 512

    # Load Cora Dataset
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    n_class = labels.max().item()+1

    model = DGI(n_features, n_hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    for epoch in range(max_epoch) :
        optimizer.zero_grad()

        # Corrupt input graph
        corrupted_idx = np.random.permutation(n_nodes)
        corrupted_features = features[corrupted_idx, :]

        target = torch.cat([torch.ones(n_nodes), torch.zeros(n_nodes)]).unsqueeze(0)
        logits = model(features, corrupted_features, adj)
        
        BCE = nn.BCEWithLogitsLoss()
        loss = BCE(logits, target)
        loss.backward()
        optimizer.step()

        print(f"Epsiode : {epoch+1} Loss : {loss.item()}")
        loss_list.append(loss)
    
    # Node Classification
    embeds = model.embed(features, adj)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_labels = labels[idx_train]
    val_labels = labels[idx_val]
    test_labels = labels[idx_test]

    accuracy = 0.0
    for _ in range(50) :
        classifier = LogisticRegression(n_hidden, n_class)
        optimizer = optim.Adam(classifier.parameters(), lr=0.01)

        for _ in range(200) :
            optimizer.zero_grad()

            logits = classifier(train_embs)

            CE = nn.CrossEntropyLoss()
            loss = CE(logits, train_labels)
            loss.backward()
            optimizer.step()
        
        logits = classifier(test_embs)
        pred_labels = torch.argmax(logits, dim=1)
        accuracy += torch.sum(pred_labels == test_labels) / test_labels.shape[0]
    accuracy /= 50

    print(f"Accuracy : {accuracy}")

    # Visualization
    style.use("ggplot")
    plt.plot(loss_list, linewidth=2.0, color="mediumpurple", label="BCEloss")
    plt.legend()
    plt.savefig("loss(normalization).png")

if __name__ == "__main__" :
    main()