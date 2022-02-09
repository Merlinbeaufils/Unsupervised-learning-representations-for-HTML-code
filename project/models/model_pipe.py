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
    def __init__(self, dataset, tree_model, node_model=None, optimizer=SGD, lengths: Tuple[int, int] = None, batch_size=64):
        super().__init__()
        if lengths is None:
            lengths = [400, 100]
        self.train_length, self.test_length = lengths
        self.optimizer = optimizer
        self.tree_model1 = tree_model
        self.tree_model2 = node_model
        self.loss_function = F.nll_loss  # keep this
        train_data, test_data = random_split(dataset, [self.train_length, self.test_length])
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        #return self.test_loader
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, train_batch):
        trees1, trees2 = train_batch
        reps1 = self.tree_model1(trees1)
        reps2 = self.tree_model2(trees2)
        return reps1, reps2
        # for tree1, tree2 in train_batch:
        #     rep1 = self.tree_model(tree1)
        #     rep2 = self.node_model(tree2) if self.node_model is not None else self.tree_model(tree2)
        #     reps1.append(rep1), reps2.append(rep2)
        # return reps1, reps2

    def training_step(self, train_batch, batch_idx):
        losses = []
        reps1, reps2 = self.forward(train_batch)
        for rep_index, rep in enumerate(reps2):
            positive = self.similarity(reps1[rep_index], rep)
            negs1 = [self.similarity(reps1[rep_index], neg_rep) for neg_rep in reps2[:rep_index]]
            negs2 = [self.similarity(reps1[rep_index], neg_rep) for neg_rep in reps2[rep_index + 1:]]
            logits = [positive] + negs1 + negs2
            losses.append(self.loss_function(Tensor(logits), torch.zeros(64).long()))
        #self.log('train_losses', loss)
        return Tensor(losses)


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
        tree_reps = super().forward(flat_trees).sum(axis=1)
        return tree_reps



class BoWSum(LightningModule):
    def __init__(self, total_vocab, vector_size=600):
        super().__init__()
        self.embedding = nn.Embedding(len(total_vocab), vector_size)

    def forward(self, tree_vec):
        return self.embedding(tree_vec)

    def training_step(self, tree_vec):
        x = self.forward(tree_vec)
        return sum(x)

    def train_dataloader(self):
        return DataLoader()

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=1e-6)
