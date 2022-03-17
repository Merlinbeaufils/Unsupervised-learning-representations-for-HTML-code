from typing import Tuple

import torch
from torch.functional import F
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy, recall

from project.models.bow_models import FlatSumBow
from project.models.pipe_transformer import SubModelTransformer
from project.models.recurrent_models import SimpleRnn, SubModelLstm
from project.pretraining_model_pipe import NoOptimizer, NoNodeModel

MAX_DEPTH=40

class TreeClassifier(LightningModule):
    """
        Takes in a pretrained tree encoder trained with contrastive learning.

        Finetunes on new classification task
        """
    def __init__(self, model_path, dataset, embedding_dim, num_labels, optimizer_type,
                 ratio, num_cpus, model_config, batch_size, lr):
        super().__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.lr = lr

        print('setting model params')
        self.linear = nn.Linear(embedding_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, self.num_labels)
        self.optimizer_type = optimizer_type

        self.train_loader, self.test_loader, self.eval_loader = self.set_loaders(dataset, ratio, num_cpus)

        print('loading model...')

        # self.tree_model: nn.Module = self.set_model(model_path, model_config)

        self.tree_model = torch.load(model_path)
        # new_dict = {}
        # for key, val in checkpoint['state_dict'].items():
        #     if key.startswith('tree_model'):
        #         new_dict[key.replace('tree_model.', '')] = val
        #
        # self.tree_model.load_state_dict(new_dict)
        self.tree_model.requires_grad_(False)

    def forward(self, trees, batch_idx):
        tree_reps = self.tree_model.tree_model(trees)
        tree_labels = F.relu(self.linear(tree_reps))
        tree_labels2 = F.relu(self.linear2(tree_labels))
        tree_labels3 = F.relu(self.linear3(tree_labels2))
        return tree_labels3  # F.log_softmax(tree_labels3, dim=1)

    def training_step(self, batch, batch_idx):
        loss, acc = self.handle_batch(batch)

        self.log('train/loss', loss)
        self.log('train/acc', acc)
        print(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.handle_batch(batch)

        self.log('val/loss', loss)
        self.log('val/acc', acc)

        print(loss.item())
        print('Validation is: ', acc)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.handle_batch(batch)

        self.log('test/loss', loss)
        self.log('test/acc', acc)

        return loss

    def handle_batch(self, batch):
        features, labels = batch
        logits = self(features, 0)
        loss = F.cross_entropy(logits, labels.float())

        acc = accuracy(logits, labels)

        return loss, acc

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.eval_loader

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

    def configure_optimizers(self):
        if self.optimizer_type.lower() == 'sgd':
            optimizer = SGD(params=self.parameters(), lr=self.lr)
        else:
            raise NoOptimizer
        return optimizer

    def set_model(self, model_path, model_config):
        num_embeddings = len(self.dataset.vocabs[0])
        temp_embedding = nn.Embedding(num_embeddings  + MAX_DEPTH, self.embedding_dim)
        node_model = FlatSumBow(temp_embedding)

        if model_config == 'lstm':
            return SubModelLstm(node_model, self.embedding_dim)
        elif model_config == 'transformer':
            return SubModelTransformer(node_model, num_embeddings, self.embedding_dim)
        elif model_config == 'bow':
            return node_model
        else:
            raise NoNodeModel





