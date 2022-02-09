import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule

logger = TensorBoardLogger('tb_logs', name='my_model')
trainer = Trainer(
    gpus=1,
    logger=[logger],
    max_epochs=5
)
trainer.fit(model)
torch.manual_seed(1)


def training_step(self, batch, batch_idx):
    x, y = batch.text[0].T, batch.label
    y_hat = self(x)
    loss = self.loss_function(y_hat, y)
    return dict(
        loss=loss,
        log=dict(
            train_loss=loss
        )
    )


class NgramBow(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NgramBow, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class MyModel(LightningModule):
    def __init__(self, embedding, lstm_input_size=300, lstm_hidden_size=100, output_size=3):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size)
        self.lin = nn.Linear(lstm_hidden_size, output_size)
        self.loss_function = nn.CrossEntropyLoss()  # keep this

    def forward(self, X: torch.Tensor):
        # need to be permuted because by default X is batch first
        x = self.embedding[X].to(self.device).permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = F.elu(x.permute(1, 0, 2))
        x = self.lin(x)
        x = x.sum(dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch.text[0].T, batch.label
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return train_iter

    def test_step(self, batch, batch_idx):
        x, y = batch.text[0].T, batch.label
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return dict(
            test_loss=loss,
            log=dict(
                test_loss=loss
            )
        )

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = dict(
            test_loss=avg_loss
        )
        return dict(
            avg_test_loss=avg_loss,
            log=tensorboard_logs
        )

    def test_dataloader(self):
        return test_iter
