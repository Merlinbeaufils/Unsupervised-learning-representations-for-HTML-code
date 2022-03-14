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
    def __init__(self, node_model, vocab_length, embedding_dim=64, dropout : float = .5):
        super().__init__()
        self.node_model = node_model
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=embedding_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = embedding_dim
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
        output = self.transformer_encoder(src)
        return output[:, -1]


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