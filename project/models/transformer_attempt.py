# =============================================================================
# Libs
# =============================================================================
import multiprocessing
from typing import List, Tuple

import pytorch_lightning
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
from project.frequency import Vocabulary, build_files, build_vocabularies
from project.dataloading import BaseTreeDataset
from torch import LongTensor

# =============================================================================
# Transformer
# =============================================================================
from project.parsing import dir_to_str, strings_to_trees, pickle_dump, pickle_load, HtmlNode
from project.tree_tokenizer import TransformerTreeTokenizer


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
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, dropout=.1):
        super().__init__()

        # model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = PositionalEmbedding(embed_size, seq_len)

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
        return self.pe[:, :x.size(1)]  # x.size(1) = seq_len


# =============================================================================
# Dataset
# =============================================================================
class SentencesDataset(Dataset):
    # Init dataset
    def __init__(self, sentences, vocab, seq_len):
        dataset = self

        dataset.sentences = sentences
        dataset.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e: i for i, e in enumerate(dataset.vocab)}
        dataset.rvocab = {v: k for k, v in dataset.vocab.items()}
        dataset.seq_len = seq_len

        # special tags
        dataset.IGNORE_IDX = dataset.vocab['<ignore>']  # replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>']  # replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>']  # replacement tag for the masked word prediction task

    # fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self

        # while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1

        # ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))]  # PAD ok

        # apply random mask
        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]

        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    # return length
    def __len__(self):
        return len(self.sentences)

    # get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s]
        return s


# BaseTreeDataset(trees, vocabs, indexes_length, total, key_only)
class TransformerTreeDataset(BaseTreeDataset):
    # Init dataset
    def __init__(self, trees: List[HtmlNode], total_vocab: Vocabulary,
                 indexes_length=1000, key_only=False, max_seq_len=512):
        super().__init__(trees=trees, vocabs=[total_vocab],
                         indexes_length=indexes_length, total=True,
                         key_only=key_only, build_samples=False)

        self.vocab = total_vocab
        self.rvocab = total_vocab.reverse_vocab()
        self.max_seq_len = max_seq_len
        self.tree_tokenizer = TransformerTreeTokenizer(total_vocab=total_vocab)

        # special tags
        self.IGNORE_IDX = self.vocab['<ignore>']  # replacement tag for tokens to ignore
        self.OUT_OF_VOCAB_IDX = self.vocab['<oov>']  # replacement tag for unknown words
        self.MASK_IDX = self.vocab['<mask>']  # replacement tag for the masked word prediction task

        self.reduce_trees(100)
        self.indexes = self.build_indexes(indexes_length)
        self.build_samples(indexes=self.indexes)
        self.padding_tokens()

    def build_sample(self, tree_index_path, tree_index) -> Tuple[List, List]:
        # node: HtmlNode = self.trees[tree_index].path[tree_index_path]
        tree: HtmlNode = self.trees[tree_index]
        node: HtmlNode = tree.path[tree_index_path]
        node.mask_affected()
        tokenized_node, tokenized_tree = self.tree_tokenizer(tree)
        node.unmask_affected()
        assert len(tokenized_tree) == len(tokenized_node)
        self.tree_max = max(len(tokenized_tree), self.tree_max)
        return tokenized_node, tokenized_tree

    def padding_tokens(self) -> None:
        # for i, (node_sample, tree_sample) in enumerate(self.samples):
        #     self.samples[i] = (node_sample[-1 * self.max_seq_len:], tree_sample[-1 * self.max_seq_len:])
        #     [self.samples[i][0].append(self.IGNORE_IDX) for i in range(self.max_seq_len - len(node_sample))]
        #     [self.samples[i][1].append(self.IGNORE_IDX) for i in range(self.max_seq_len - len(tree_sample))]
        # assert len(self.samples[0][0]) == len(self.samples[0][1])
        for i, (node_sample, tree_sample) in enumerate(self.samples):
            node_sample, tree_sample = node_sample[-1 * self.max_seq_len:], tree_sample[-1 * self.max_seq_len:]
            [node_sample.append(self.IGNORE_IDX) for i in range(self.max_seq_len - len(node_sample))]
            [tree_sample.append(self.IGNORE_IDX) for i in range(self.max_seq_len - len(tree_sample))]
            self.samples[i] = node_sample, tree_sample
        assert len(self.samples[0][0]) == len(self.samples[0][1])

    # fetch data
    def __getitem__(self, index):
        node_sample, tree_sample = self.samples[index]
        return LongTensor(node_sample), LongTensor(tree_sample)

    # return length
    def __len__(self):
        return len(self.samples)


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.vocab) + MAX_DEPTH, seq_len, dropout)
        self.configure_optimizers()
        self.loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_data, self.test_data = random_split(dataset, [train_size, test_size])

    def configure_optimizers(self):
        optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)}
        return optim.Adam(self.model.parameters(), **optim_kwargs)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        kwargs = {'num_workers': multiprocessing.cpu_count(), 'shuffle': True, 'drop_last': True,
                  'pin_memory': True,
                  'batch_size': batch_size}
        return torch.utils.data.DataLoader(self.train_data, **kwargs)

    def test_dataloader(self):
        kwargs = {'num_workers': multiprocessing.cpu_count(), 'shuffle': True, 'drop_last': True,
                  'pin_memory': True,
                  'batch_size': batch_size}
        return torch.utils.data.DataLoader(self.test_data, **kwargs)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        output = self.forward(features, batch_idx)

        output_v = output.view(-1, output.shape[-1])
        target_v = labels.view(-1, 1).squeeze()
        loss = self.loss_model(output_v, target_v)
        return loss

    def forward(self, features, batch_idx):
        return self.model(features)


if __name__ == '__main__':
    # =============================================================================
    # #Init
    # =============================================================================
    print('initializing..')
    batch_size = 32
    seq_len = 512
    embed_size = 128
    inner_ff_size = embed_size * 4
    n_heads = 8
    n_code = 8
    dropout = 0.1
    MAX_DEPTH = 40
    only_keys = False
    using_gpu = False

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
                                     key_only=only_keys, max_seq_len=seq_len)

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



    transformer_model = MyModel()
    if using_gpu:
        gpus = 1
        transformer_model = transformer_model.cuda()
    else:
        gpus = 0
    logger = TensorBoardLogger('tb_logs', name='transformer')
    trainer = Trainer(
        gpus=0,
        logger=[logger],
        max_epochs=5
    )
    trainer.fit(transformer_model)
    # =============================================================================
    # Results analysis
    # =============================================================================
    print('saving embeddings...')
    # N = 3000
    # np.savetxt('values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    # s = [dataset.rvocab[i] for i in range(N)]
    # open('names.tsv', 'w+').write('\n'.join(s))

    print('end')
