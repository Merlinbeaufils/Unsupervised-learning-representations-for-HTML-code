import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


class SubModelLstm(nn.Module):
    """
    Applies a sub-model at node level then inputs the node representations into the lstm.
    Currently, the node model is bow
    """
    def __init__(self, node_model, hidden_size=60):
        super().__init__()
        self.node_model = node_model
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=self.node_model.embedding.embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True)

    def forward(self, trees) -> Tensor:
        """

        :param trees: batch of tensors of shape (batch_size, tree_len_max, node_len_max)
        :param tree_lens: tree_lens, maximum value of tree_len_max
        :return: tensor of shape (batch_size, embedding_dim)
        """
        self.hidden = self.init_hidden(trees)
        tree_reps = self.node_model(trees, node=False)

        tree_lens = (torch.sum(tree_reps, dim=2) != 0).long().argmin(dim=1)
        tree_lens = (tree_lens == 0).float() * trees.shape[1] + tree_lens

        packed = torch.nn.utils.rnn.pack_padded_sequence(tree_reps, tree_lens, batch_first=True, enforce_sorted=False)
        packed, self.hidden = self.lstm(packed, self.hidden)
        # unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        return self.hidden[0].view(-1, self.hidden_size)

    def init_hidden(self, trees):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, trees.shape[0], self.hidden_size, device=trees.device)
        hidden_b = torch.randn(1, trees.shape[0], self.hidden_size, device=trees.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return hidden_a, hidden_b

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
