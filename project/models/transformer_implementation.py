# =============================================================================
# Libs
# =============================================================================

import math
from os.path import exists

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset, random_split

from project.dataloading import TransformerTreeDataset
from project.frequency import build_files, build_vocabularies
# =============================================================================
# Transformer
# =============================================================================
from project.parsing import dir_to_str, strings_to_trees, pickle_dump, pickle_load


def attention(q, k, v, mask=None, dropout=None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])

    # mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

        #        self.q_linear = nn.Linear(out_dim, out_dim)
        #        self.k_linear = nn.Linear(out_dim, out_dim)
        #        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim * 3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, x, y=None, mask=None):
        # in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y

        qkv = self.linear(x)  # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim]  # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim * 2]  # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim * 2:]  # BS * SEQ_LEN * EMBED_SIZE_L

        # break into n_heads
        q, k, v = [self.split_heads(t) for t in (q, k, v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD

        # n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout)  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1, 2).contiguous().view(scores.shape[0], -1,
                                                          self.out_dim)  # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE

        return out


class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


class Transformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, max_seq_len, dropout=.1):
        super().__init__()

        # model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = PositionalEmbedding(embed_size, max_seq_len)

        # backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        # language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings, bias=False)

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # x.size(1) = max_seq_len


class TransformerModule(LightningModule):
    def __init__(self, dataset, kwargs=None, optim_kwargs=None, loader_kwargs=None):
        # embed_size must be divisible by n_heads
        self.kwargs = {'n_code': 8, 'n_heads': 8, 'embed_size': 64, 'inner_ff_size': 240,
                       'n_embeddings': len(dataset.vocab) + 40,
                       'max_seq_len': 512, 'dropout': 0.1} \
                    if kwargs is None else kwargs
        self.optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)} \
                    if optim_kwargs is None else optim_kwargs

        self.loader_kwargs = {'num_workers': 2, 'shuffle': True, 'drop_last': True,
                              'pin_memory': True, 'batch_size': 64} \
                    if loader_kwargs is None else loader_kwargs

        super().__init__()
        self.model = Transformer(**kwargs)
        self.loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_data, self.test_data = random_split(dataset, [train_size, test_size])

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), **self.optim_kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, **self.loader_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, **self.loader_kwargs)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        output = self.forward(features, batch_idx)

        output_v = output.view(-1, output.shape[-1])
        target_v = labels.view(-1, 1).squeeze()
        loss = self.loss_model(output_v, target_v)
        print(loss)
        self.log("train/loss", loss)
        return loss

    def forward(self, features, batch_idx):
        return self.model(features)

    # def validation_step(self, batch, batch_idx):
    #     features, labels = batch
    #     output = self.forward(features, batch_idx)
    #
    #     output_v = output.view(-1, output.shape[-1])
    #     target_v = labels.view(-1, 1).squeeze()
    #     loss = self.loss_model(output_v, target_v)
    #     print(loss)
    #     self.log("val/loss", loss)
    #     self.log("val/accuracy", accuracy)
    #
    #     print('Validation: ', accuracy)
    #     return loss


if __name__ == '__main__':
    # =============================================================================
    # #Init
    # =============================================================================
    print('initializing..')

    # =============================================================================
    # Input
    # =============================================================================
    # 1) load files into trees
    setup_location = './data/common_sites/'
    print('loading files...')
    trees = strings_to_trees(dir_to_str(setup_location))
    build_files(setup_location, setup_location + 'text_files')

    # 2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    # print('tokenizing sentences...')
    # special_chars = ',?;.:/*!+-()[]{}"\'&'
    # sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    # sentences = [[w for w in s if len(w)] for s in sentences]

    # 3) create vocab if not already created
    print('creating/loading vocab...')
    directory = setup_location + 'vocabs/total'
    if not exists(directory):
        vocab = build_vocabularies(setup_location)[-1]
        pickle_dump(setup_location + 'vocabs/total', vocab)
    else:
        vocab = pickle_load(directory)
    vocab.__setitem__('<son>', len(vocab))
    vocab.__setitem__('<eon>', len(vocab))

    # 4) create dataset
    print('creating dataset...')
    dataset = TransformerTreeDataset(trees=trees, total_vocab=vocab, indexes_length=200,
                                     key_only=True, max_seq_len=512)

    # =============================================================================
    # Model
    # =============================================================================
    # init model
    print('initializing model...')

    # =============================================================================
    # Optimizer
    # =============================================================================
    # print('initializing optimizer and loss...')
    # optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    # loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

    # =============================================================================
    # Train
    # =============================================================================
    print('training...')

    # =============================================================================
    # Results analysis
    # =============================================================================
    print('saving embeddings...')
    # N = 3000
    # np.savetxt('values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    # s = [dataset.rvocab[i] for i in range(N)]
    # open('names.tsv', 'w+').write('\n'.join(s))

    print('end')
