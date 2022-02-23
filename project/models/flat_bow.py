import torch
import torch.nn as nn


class FlatSum(nn.Module):
    def __init__(self, embedding, mean=False, node=False):
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


class FlatSumNew(nn.Module):
    # specify node to deal with node or flatten. Else keeps tree dimension.
    def __init__(self, embedding, mean=False):
        super().__init__()
        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.mean = mean

    def forward(self, trees, node=True):  # (batch_size, tree_size, node_size) or (batch_size, node_size)
        tree_size = 1 if node and len(trees.shape) == 2 else trees.shape[1]
        batch_size, node_size = trees.shape[0], trees.shape[-1]

        flat_trees = torch.flatten(trees, start_dim=1).long()  # (batch_size, tree_size x node_size)
        flat_tree_reps = self.embedding(flat_trees)   # (batch_size, tree_size x node_size, embedding_dim)

        mask = flat_trees == 0
        flat_tree_reps[mask] = 0

        tree_reps = flat_tree_reps.reshape(batch_size, tree_size, node_size, self.embedding_dim)
        # if node return (batch_size, embed_dim) else return (batch_size, tree_size, embed_dim)
        # if mean return mean else return sum
        dims = [1, 2] if node else [2]
        if self.mean:
            tree_reps = tree_reps.mean(dim=dims)
        else:
            tree_reps = tree_reps.sum(dim=dims)

        return tree_reps


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

