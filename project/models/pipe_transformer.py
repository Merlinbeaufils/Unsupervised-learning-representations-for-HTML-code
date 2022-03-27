import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class SubModelTransformer(nn.Module):
    """
    Applies a sub-model at node level then inputs the node representations into the transformer.
    Currently, the node model is bow
    """
    def __init__(self, node_model, seq_len, embedding_dim=64, dropout : float = .5, nheads=4,
                 num_layers=2, config='last'):
        super().__init__()
        self.node_model = node_model
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads, dim_feedforward=embedding_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)  # COME BACK TO THIS
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.config = config

        if config == 'linear':
            self.linear = nn.Linear(seq_len * embedding_dim, embedding_dim)
        # self.decoder = nn.Linear(d_model, ntoken)

    # def forward(self, trees) -> Tensor:
    #     tree_reps = self.node_model(trees, node=False)
    #     tree_reps_final, state = self.transformer(tree_reps,)
    #     return tree_reps_final[:, -1]

    def forward(self, trees) -> Tensor:
        """
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        tree_reps = self.node_model(trees, node=False)
        src = self.pos_encoder(tree_reps)
        output = self.transformer_encoder(src)  # (batch_size, seq_len, embedding_dim)
        if self.config == 'last':
            new_output = output[:, -1]
        elif self.config == 'mean':
            new_output = torch.mean(output, dim=1)
        elif self.config == 'linear':
            flat_output = output.reshape(output.shape[0], -1)
            new_output = self.linear(flat_output)
        else:
            raise WrongTransformerConfig
        return new_output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class WrongTransformerConfig(Exception):
    pass