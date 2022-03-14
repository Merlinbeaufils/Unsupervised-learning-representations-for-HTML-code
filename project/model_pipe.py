from typing import Tuple

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from project.models.bow_models import FlatSumBow
from project.models.pipe_transformer import SubModelTransformer
from project.models.recurrent_models import SimpleRnn, SubModelLstm

MAX_DEPTH = 40


class BaseModel(LightningModule):
    """
    Applies contrastive learning framework on the "node" and "tree" models.
    These are just like context and masked label models.

    Standard functions are:
        similarity: cosine
        loss: cross entropy
        optimizer: adam or sgd
    """
    def __init__(self, dataset, node_model_type, vocab_size,
                 tree_model_type=None, optimizer_type='sgd',
                 batch_size=10, lr=1e-3, loss_type='cross_entropy',
                 similarity_type='cosine', embedding_dim=60,
                 train_proportion=0.8, num_cpus=2):
        super().__init__()
        self.dataset = dataset
        self.tree_model_type = tree_model_type
        self.node_model_type = node_model_type
        self.optimizer_type = optimizer_type
        self.similarity_type = similarity_type
        self.batch_size = batch_size
        self.lr = lr
        self.loss_type = loss_type
        self.train_proportion = train_proportion

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size + MAX_DEPTH, embedding_dim=embedding_dim)
        self.loss_function = self.configure_loss()
        self.similarity = self.configure_similarity()
        self.node_model = self.set_node_model()
        self.tree_model = self.set_tree_model()

        self.train_loader, self.eval_loader, self.test_loader = \
            self.set_loaders(dataset, self.train_proportion, num_cpus)

    def forward(self, train_batch) -> Tuple[Tensor, Tensor]:
        trees1, trees2 = train_batch
        reps1 = self.node_model(trees1)
        reps2 = self.tree_model(trees2)
        return reps1, reps2

    def training_step(self, train_batch, batch_idx):
        reps1, reps2 = self.forward(train_batch)  # batch_dim x embedding_dim
        # reps1.norm(dim=1)
        # reps1 = reps1 / torch.linalg.norm(reps1, dim=1, keepdims=True)
        # reps2 = reps2 / torch.linalg.norm(reps2, dim=1, keepdims=True)

        scores = reps1 @ reps2.T  # batch_dim x batch_dim
        labels = torch.arange(reps1.size(0), device=reps1.device)  # batch_dim x batch_dim
        loss = self.loss_function(scores, labels)

        print(loss.item())
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        reps1, reps2 = self.forward(batch)
        scores = reps1 @ reps2.T

        labels = torch.arange(reps1.size(0), device=reps1.device)  # batch_dim x batch_dim
        loss = self.loss_function(scores, labels)

        accuracy_vector = (torch.argmax(scores, dim=1) == labels).float()
        accuracy = accuracy_vector.mean()

        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)

        print('Validation: ', accuracy)
        return loss  # CHANGE

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.eval_loader

    def configure_optimizers(self):
        if self.optimizer_type.lower() == 'sgd':
            optimizer = SGD(params=self.parameters(), lr=self.lr)
        else:
            raise NoOptimizer
        return optimizer

    def configure_loss(self):
        if self.loss_type.lower() == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss()
        else:
            raise NoLoss
        return loss_function

    def set_loaders(self, dataset, ratio, num_cpus):
        train_size = int(ratio * len(dataset))
        eval_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - eval_size
        train_data, eval_data, test_data = random_split(dataset, [train_size, eval_size, test_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=num_cpus)
        eval_loader = DataLoader(eval_data, batch_size=self.batch_size,
                                 num_workers=num_cpus)
        test_loader = DataLoader(test_data, batch_size=self.batch_size,
                                 num_workers=num_cpus)
        return train_loader, eval_loader, test_loader

    def configure_similarity(self):
        if self.similarity_type == 'cosine':
            similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        else:
            raise NoSimilarity
        return similarity

    def set_node_model(self):
        name = self.node_model_type
        if name == 'bow':
            node_model = FlatSumBow(self.embedding)
        elif name == 'rnn':
            node_model = SimpleRnn(self.embedding, self.inner_dim)
        elif name == 'lstm':
            node_model = SubModelLstm(FlatSumBow(self.embedding), hidden_size=self.embedding_dim)
        elif name == 'transformer':
            node_model = SubModelTransformer(FlatSumBow(self.embedding), vocab_length=len(self.dataset.vocab))
        else:
            raise NoNodeModel
        return node_model

    def set_tree_model(self):
        name = self.tree_model_type
        if name is None:
            tree_model = self.node_model
        elif name == 'flat':
            tree_model = FlatSumBow(self.embedding)
        elif name == 'lstm':
            tree_model = SubModelLstm(self.node_model, hidden_size=self.embedding_dim)
        elif name == 'transformer':
            tree_model = SubModelTransformer(self.node_model, vocab_length=len(self.dataset.vocab))
        else:
            raise NoTreeModel
        return tree_model


class NoOptimizer(Exception):
    pass


class NoLoss(Exception):
    pass


class NoSimilarity(Exception):
    pass


class NoTreeModel(Exception):
    pass


class NoNodeModel(Exception):
    pass

############################################################################
#            Not Used                                                      #
############################################################################


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
