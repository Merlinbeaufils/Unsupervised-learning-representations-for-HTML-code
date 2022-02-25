import torch
import torch.nn as nn


class FlatSumBow(nn.Module):
    """
    Bag of words implementation.
    Returns sum or mean of vector embeddings of indexes given.

    Possible shape changes:
    (batch_size, node_size) -> (batch_size, node_size, embedding_dim) if node=True
    (batch_size, tree_size, node_size) -> (batch_size, tree_size * node_size, embedding_dim) if node=True
    (batch_size, tree_size, node_size) -> (batch_size, tree_size, node_size, embedding_dim) if node=False

    Basically, when treating tree, you can choose to preserve outer dim by specifying node=False
    """

    # specify node to deal with node or flatten. Else keeps tree dimension.
    def __init__(self, embedding: nn.Embedding, mean: bool = False):
        super().__init__()
        self.embedding:     nn.Embedding = embedding
        self.embedding_dim: int = embedding.embedding_dim
        self.mean:          bool = mean

    def forward(self, trees, node=True):  # (batch_size, tree_size, node_size) or (batch_size, node_size)
        # keep desired shape before flattening
        tree_size = 1 if node and len(trees.shape) == 2 else trees.shape[1]
        batch_size, node_size = trees.shape[0], trees.shape[-1]

        # flatten and embed trees
        flat_trees = torch.flatten(trees, start_dim=1).long()  # (batch_size, tree_size x node_size)
        flat_tree_reps = self.embedding(flat_trees)   # (batch_size, tree_size x node_size, embedding_dim)

        # reset pad values to 0
        mask = flat_trees == 0
        flat_tree_reps[mask] = 0

        # reshape to desired shape
        tree_reps = flat_tree_reps.reshape(batch_size, tree_size, node_size, self.embedding_dim)

        # if node return (batch_size, embed_dim) else return (batch_size, tree_size, embed_dim)
        dims = [1, 2] if node else [2]
        # if mean return mean else return sum
        if self.mean:
            tree_reps = tree_reps.mean(dim=dims)
        else:
            tree_reps = tree_reps.sum(dim=dims)

        return tree_reps

#############################################################################################
#  Not in USE                                                                               #
#############################################################################################


class FlatEmbedding(nn.Embedding):
    def forward(self, trees):
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = super().forward(flat_trees)
        mask = flat_trees == 0
        tree_reps[mask] = 0
        # tree_reps = tree_reps.mean(dim=1)
        tree_reps = tree_reps.sum(dim=1)
        return tree_reps


class FlatEmbeddingAndLinear(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.weight.requires_grad = False

    def forward(self, trees):
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = super().forward(flat_trees)
        tree_reps = self.linear(tree_reps)
        tree_reps = tree_reps.sum(dim=1)
        return tree_reps


class SimpleLinear(nn.Linear):
    def forward(self, trees):
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = super().forward(flat_trees).sum(dim=1)
        return tree_reps


class FlatSum(nn.Module):
    def __init__(self, embedding, mean=False):
        super().__init__()
        self.embedding = embedding
        self.mean = mean

    def forward(self, trees):
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = self.embedding(flat_trees)
        mask = flat_trees == 0
        tree_reps[mask] = 0
        if self.mean:
            tree_reps = tree_reps.mean(dim=1)
        tree_reps = tree_reps.sum(dim=1)
        return tree_reps
