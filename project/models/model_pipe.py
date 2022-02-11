import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, random_split
from torch import Tensor


class BaseModel(LightningModule):
    def __init__(self, dataset, tree_model, node_model=None, optimizer=SGD, batch_size=10, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.tree_model = tree_model
        self.node_model = tree_model if node_model is None else node_model
        self.loss_function = torch.nn.CrossEntropyLoss()  # change this in a sec
        self.activation = torch.nn.ReLU()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, train_batch):
        trees1, trees2 = train_batch
        reps1 = self.node_model(trees1)
        reps2 = self.tree_model(trees2)
        return reps1, reps2

    def training_step(self, train_batch, batch_idx):
        reps1, reps2 = self.forward(train_batch)  # 64x600

        reps1 = reps1 / torch.linalg.norm(reps1, dim=1, keepdims=True)
        reps2 = reps2 / torch.linalg.norm(reps2, dim=1, keepdims=True)
        # reps1, reps2 = self.activation(reps1, inplace=True), self.activation(reps2, inplace=True)


        scores = reps1 @ reps2.T
        labels = torch.eye(reps1.size(0), device=reps1.device)
        loss = self.loss_function(scores, labels)
        return loss


def random_stuff():
    tree_tensor4 = torch.stack(
        [Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5])])
    tree_tensor1 = torch.stack(
        [Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5])])
    tree_tensor2 = torch.stack(
        [Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5])])
    tree_tensor3 = torch.stack(
        [Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5]), Tensor([1, 2, 3, 4, 5])])
    batch = torch.stack([tree_tensor2, tree_tensor3, tree_tensor1, tree_tensor4])
    torch.flatten(batch, start_dim=1)
    print('hello')


#random_stuff()


class FlatEmbedding(nn.Embedding):
    def forward(self, trees) -> Tensor:
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = super().forward(flat_trees)
        tree_reps = tree_reps.sum(dim=1)
        return tree_reps


class FlatEmbeddingAndLinear(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.weight.requires_grad = False

    def forward(self, trees) -> Tensor:
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = super().forward(flat_trees)
        tree_reps = self.linear(tree_reps)
        tree_reps = tree_reps.sum(dim=1)
        return tree_reps


class SimpleLinear(nn.Linear):
    def forward(self, trees) -> Tensor:
        flat_trees = torch.flatten(trees, start_dim=1).long()
        tree_reps = super().forward(flat_trees).sum(dim=1)
        return tree_reps

