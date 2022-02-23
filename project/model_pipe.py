import multiprocessing

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from project.models.flat_bow import FlatEmbedding, FlatSum, FlatEmbeddingAndLinear, FlatSumNew
from project.models.RNNs import SimpleRnn, SimpleLSTM, StackedRnn, SubModelLstm

MAX_DEPTH = 40


class BaseModel(LightningModule):
    def __init__(self, dataset, node_model_type, vocab_size,
                 tree_model_type=None, optimizer_type='sgd',
                 batch_size=10, lr=1e-3, loss_type='cross_entropy',
                 similarity_type='cosine', embedding_dim=600, train_proportion=0.8):
        super().__init__()
        self.tree_model_type = tree_model_type
        self.node_model_type = node_model_type
        self.optimizer_type = optimizer_type
        self.similarity_type = similarity_type
        self.batch_size = batch_size
        self.lr = lr
        self.loss_type = loss_type
        self.train_proportion = train_proportion

        self.embedding = nn.Embedding(num_embeddings=vocab_size + MAX_DEPTH, embedding_dim=embedding_dim)
        self.loss_function = self.configure_loss()
        self.similarity = self.configure_similarity()
        self.tree_model, self.node_model = self.configure_submodels()

        self.train_loader, self.eval_loader, self.test_loader = self.set_loaders(dataset, self.train_proportion)

    def forward(self, train_batch):
        trees1, trees2 = train_batch
        reps1 = self.node_model(trees1)
        reps2 = self.tree_model(trees2)
        return reps1, reps2

    def training_step(self, train_batch, batch_idx):
        reps1, reps2 = self.forward(train_batch)  # batch_dim x embedding_dim
        #reps1 = reps1 / torch.linalg.norm(reps1, dim=1, keepdims=True)
        #reps2 = reps2 / torch.linalg.norm(reps2, dim=1, keepdims=True)

        scores = reps1 @ reps2.T  # batch_dim x batch_dim
        labels = torch.arange(reps1.size(0), device=reps1.device)  # batch_dim x batch_dim
        loss = self.loss_function(scores, labels)
        print(loss.item())
        return loss

    # def validation_step(self, batch, batch_idx):
    #     reps1, reps2 = self.forward(batch)
    #     scores = torch.diag(reps1 @ reps2.T)
    #     return scores.mean()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return self.eval_loader

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

    def set_loaders(self, dataset, ratio):
        train_size = int(ratio * len(dataset))
        eval_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - eval_size
        train_data, eval_data, test_data = random_split(dataset, [train_size, eval_size, test_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=multiprocessing.cpu_count())
        eval_loader = DataLoader(eval_data, batch_size=self.batch_size,
                                 num_workers=multiprocessing.cpu_count())
        test_loader = DataLoader(test_data, batch_size=self.batch_size,
                                 num_workers=multiprocessing.cpu_count())
        return train_loader, eval_loader, test_loader

    def configure_similarity(self):
        if self.similarity_type == 'cosine':
            similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        else:
            raise NoSimilarity
        return similarity

    def configure_submodels(self):
        if self.node_model_type == 'flat':
            node_model = FlatSumNew(self.embedding)
        elif self.node_model_type == 'rnn':
            node_model = SimpleRnn(self.embedding, self.embedding.embedding_dim)
        else:
            raise NoTreeModel

        if self.tree_model_type is None:
            tree_model = node_model
        elif self.tree_model_type == 'flat':
            tree_model = StackedRnn(node_model, self.embedding, 0)
        elif self.tree_model_type == 'lstm':
            tree_model = SubModelLstm(node_model, hidden_size=600)
        else:
            raise NoNodeModel

        return tree_model, node_model



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

