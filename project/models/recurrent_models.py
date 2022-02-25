import torch
import torch.nn as nn
from torch import Tensor


class SubModelLstm(nn.Module):
    """
    Applies a sub-model at node level then inputs the node representations into the lstm.
    Currently, the node model is bow
    """
    def __init__(self, node_model, hidden_size=60):
        super().__init__()
        self.node_model = node_model
        self.lstm = nn.LSTM(
            input_size=self.node_model.embedding.embedding_dim,
            hidden_size=hidden_size,
            batch_first=True)

    def forward(self, trees) -> Tensor:
        tree_reps = self.node_model(trees, node=False)
        tree_reps_final, state = self.lstm(tree_reps)
        return tree_reps_final[:, -1]
################################################################################
#     Not Used                                                                 #
################################################################################


class StackedRnn(nn.Module):
    def __init__(self, node_model, hidden_size):
        super().__init__()
        self.node_model = node_model
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(self, trees) -> Tensor:
        tree_reps = torch.cat([self.node_model(tree) for tree in trees])
        mask = trees == 0
        tree_reps[mask] = 0
        tree_reps_final, state = self.rnn(tree_reps)
        final = tree_reps_final[:, -1]
        return final


class SimpleLSTM(nn.Module):
    def __init__(self, embedding, hidden_size=60):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            batch_first=True)

    def forward(self, trees) -> Tensor:
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = self.embedding(flat_trees)
        mask = flat_trees == 0
        tree_reps[mask] = 0
        tree_reps_final, state = self.rnn(tree_reps)
        final = tree_reps_final[:, -1]
        return final


class SimpleRnn(nn.Module):
    def __init__(self, embedding, hidden_size=60):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.RNN(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            batch_first=True)

    def forward(self, trees) -> Tensor:
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = self.embedding(flat_trees)
        mask = flat_trees == 0
        tree_reps[mask] = 0
        tree_reps_final, state = self.rnn(tree_reps)
        return tree_reps_final[:, -1]
        # final = tree_reps_final[:, -1]
        # return final
