# from typing import List, Tuple
import random
from typing import Tuple, List

import pandas
from torch.functional import F
from torch import Tensor, LongTensor
from torch.utils.data import Dataset

from project.sparsing import length_depth_reduction
from project.frequency import Vocabulary
from project.parsing import HtmlNode, string_to_tree
from project.sparsing import random_sparse
from project.tree_tokenizer import Node_Tokens, Tree_Tokens, BaseTokenizer, KeyOnlyTokenizer, TreeTokenizer, \
    TransformerTreeTokenizer, NoKeysTokenizer

Sample = Tuple[Node_Tokens, Tree_Tokens]
Samples = List[Sample]
TensorizedSample = Tuple[Tensor, Tensor]
Forest = List[HtmlNode]
PAD_VALUE = 0


class UnknownSampleConfig(Exception):
    pass


class BaseTreeDataset(Dataset):  # Tree dataset class allowing handling of html trees
    """
    Dataset to deal with building samples from HtmlNode class

    Indexes cover every possible subtree-tree combination
    Samples are built before runtime

    build_sample function defines how samples are build from the indexes given
    """
    def __init__(self, trees: List[HtmlNode], vocabs: List[Vocabulary],
                 indexes_length=1000, total: bool = False, key_only: bool = False,
                 build_samples: bool = True, index_config='per_tree', per_tree=10,
                 sample_config='base', no_keys=True):
        # indexes is a list of (tree_path_index, tree_index) tuples indicating (node, tree)
        super().__init__()
        self.trees:   Forest = trees
        self.samples = []
        self.total = total
        self.vocab = vocabs[0]
        self.key_only = key_only
        self.no_keys = no_keys
        self.node_tokenizer, self.tree_tokenizer = self.set_tokenizers(vocabs=vocabs)
        self.index_builder: BaseIndexBuilder = self.config_index_builder(indexes_length, index_config, per_tree)
        self.sample_builder: BaseSampleBuilder = self.config_sample_builder(sample_config)
        if build_samples:
            self.indexes = self.index_builder.build_indexes()
            self.samples = self.sample_builder.build_samples(self.indexes)
            self.padding_tokens(sample_config)

    def __getitem__(self, index: int) -> TensorizedSample:  # returns a (subtree, tree) pair. Tokenized
        token_node, token_tree = self.samples[index]
        return LongTensor(token_node), LongTensor(token_tree)

    def __len__(self) -> int:  # returns # of samples
        return len(self.indexes)

    def config_index_builder(self, max_indexes, index_config, per_tree):
        if index_config == 'all':
            return BaseIndexBuilder(self.trees, max_indexes)
        elif index_config == 'per_tree':
            return PerTreeIndexBuilder(self.trees, max_indexes, per_tree)

    def handle_index(self, indexes, tree_path_index, tree_index):
        indexes.append((tree_path_index, tree_index))

    def config_sample_builder(self, sample_config):
        if sample_config == 'base':
            return BaseSampleBuilder(self.trees, self.node_tokenizer, self.tree_tokenizer)
        elif sample_config == 'transformer':
            return TransformerSampleBuilder(self.trees, self.node_tokenizer, self.tree_tokenizer)

    def padding_tokens(self, sample_config) -> None:
        print('padding tokens...')
        for node_sample, tree_sample in self.samples:
            if sample_config == 'transformer':
                # print('~~~~~~~~~~~~~~ Working correctly ~~~~~~~~~~~~')
                pad_tree(node_sample, self.sample_builder.tree_max, self.sample_builder.node_max)
            elif sample_config == 'base':
                pad_node(node_sample, self.sample_builder.node_max)
            else:
                raise UnknownSampleConfig

            pad_tree(tree_sample, self.sample_builder.tree_max, self.sample_builder.node_max)

    def reduce_trees(self, max_size=500):
        for tree in self.trees:
            random_sparse(tree, max_size)

    def set_tokenizers(self, vocabs: List[Vocabulary]):
        if self.no_keys:
            node_tokenizer = NoKeysTokenizer(vocabs=vocabs, total=self.total)
        elif self.key_only:
            node_tokenizer = KeyOnlyTokenizer(vocabs=vocabs, total=self.total)
        else:
            node_tokenizer = BaseTokenizer(vocabs=vocabs, total=self.total)
        tree_tokenizer = TreeTokenizer(vocabs=vocabs, total=self.total, key_only=self.key_only, no_keys=self.no_keys)
        return node_tokenizer, tree_tokenizer


class ContTreeDataset(BaseTreeDataset):  # Samples are of type: (masked_tree, tree)
    def build_sample(self, node_indexes, tree_index):
        tree: HtmlNode = self.trees[tree_index]
        tokenized_tree = self.tree_tokenizer(tree)
        [tree.path[i].mask_self() for i in node_indexes]
        masked_tree = self.tree_tokenizer(tree)
        [tree.path[i].unmask_self() for i in node_indexes]
        tree_node_max = len(max(tokenized_tree, default=0, key=len))
        self.node_max = max(tree_node_max, self.node_max)
        self.tree_max = max(len(tokenized_tree), self.tree_max)
        return masked_tree, tokenized_tree

    def handle_index(self, indexes, tree_path_index, tree_index):
        indexes.append([tree_path_index], tree_index)

    def padding_tokens(self) -> None:
        for masked_tree, tree in self.samples:
            pad_tree(tree, self.tree_max, self.node_max)
            pad_tree(masked_tree, self.tree_max, self.node_max)


class TransformerTreeDataset(BaseTreeDataset):
    # Init dataset
    def __init__(self, trees: List[HtmlNode], total_vocab: Vocabulary,
                 indexes_length=1000, key_only=False, max_seq_len=512,
                 index_config='per_tree', per_tree=10):

        super().__init__(trees=trees, vocabs=[total_vocab],
                         indexes_length=indexes_length, total=True,
                         key_only=key_only, build_samples=False, index_config=index_config,
                         per_tree=per_tree)

        self.vocab = total_vocab
        self.rvocab = total_vocab.reverse_vocab()
        self.max_seq_len = max_seq_len
        self.tree_tokenizer = TransformerTreeTokenizer(total_vocab=total_vocab)

        # special tags
        self.IGNORE_IDX = self.vocab['<ignore>']  # replacement tag for tokens to ignore
        self.OUT_OF_VOCAB_IDX = self.vocab['<oov>']  # replacement tag for unknown words
        self.MASK_IDX = self.vocab['<mask>']  # replacement tag for the masked word prediction task

        self.reduce_trees(100)
        self.indexes = self.index_builder.build_indexes()
        self.build_samples(indexes=self.indexes)
        self.padding_tokens(0)

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

    def padding_tokens(self, ignore) -> None:
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


class TreeClassifierDataset(Dataset):
    def __init__(self, panda_file_path, num_samples, vocabs, total=True, key_only=False, no_keys=False):
        super(TreeClassifierDataset, self).__init__()
        self.samples = []
        self.vocabs = vocabs
        self.total = total
        self.key_only = key_only
        self.no_keys = no_keys

        self.tree_tokenizer = self.set_tokenizer()

        self.tree_node_max = 0
        self.tree_max = 0

        pandas_file = pandas.read_feather(panda_file_path)
        pandas_file = pandas_file.sample(frac=1)

        self.num_samples = num_samples
        labels = set(pandas_file['tld'])
        labels = [x for x in labels]
        self.labels_to_int = {labels[i]: i for i in range(len(labels))}
        self.num_labels = len(set(pandas_file['tld']))
        self.build_samples(pandas_file)
        self.pad_samples()


    def __getitem__(self, item):
        sample = self.samples[item]
        feature = Tensor(sample[0])
        label = F.one_hot(Tensor([sample[1]]).long(), num_classes=self.num_labels).view(self.num_labels)
        return feature, label

    def __len__(self):
        return len(self.samples)

    def build_samples(self, pandas_file):
        i = 0
        print('building samples...')
        for element in pandas_file.iterrows():
            if len(self.samples) > self.num_samples:
                return
            i += 1
            string, label = element[1]['html'], element[1]['tld']

            try:
                tree = self.build_tree(string)
                label = self.labels_to_int[label]
                self.samples.append((tree, label))
                print(i)
            except Exception:
                pass

    def build_tree(self, string_bytes):
        string = string_bytes.decode('UTF-8', errors='ignore')
        tree = string_to_tree(string)
        tree = length_depth_reduction(tree, 500, 5)
        tree_token = self.tree_tokenizer(tree)

        # update max sizes
        self.tree_node_max = max(self.tree_node_max, len(max(tree_token, default=0, key=len)))
        self.tree_max = max(len(tree_token), self.tree_max)

        return tree_token

    def set_tokenizer(self):
        return TreeTokenizer(self.vocabs, self.total, self.key_only)

    def pad_samples(self):
        print('padding samples...')
        for sample in self.samples:
            tree = sample[0]
            pad_tree(tree, self.tree_max, self.tree_node_max)












def pad_tree(tree: List, length_tree: int, length_node: int) -> None:
    for node in tree:
        pad_node(node, length_node)
    while len(tree) < length_tree:
        tree.append([PAD_VALUE] * length_node)
    if len(tree) > length_tree:
        print('This is the expected tree length:', length_tree)
        print('This is the actual tree length:', len(tree))
        raise PadError


def pad_node(node: List, length_node: int) -> None:
    while len(node) < length_node:
        node.append(PAD_VALUE)
    if len(node) > length_node:
        print('This is the expected node length:', length_node)
        print('This is the actual node length:', len(node))
        raise PadError


'''
def basic_data_loader_build(args: Namespace, size: int = 500) -> (DataLoader, DataLoader):
    indexes = total_build_indexes(size)
    train_length = int(0.8 * len(indexes))
    # test_length = len(indexes) - train_length
    training_data = BaseTreeDataset(indexes[:train_length], './data/common_sites/trees/trees_short', args)
    test_data = BaseTreeDataset(indexes[train_length:], './data/common_sites/trees/trees_short', args)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader
'''
# dataset = BaseTreeDataset(indexes, './tree_directory')


def collate_function(batch: Samples):
    node_max, tree_max = 0, 0
    new_batch = []
    for node, tree in batch:
        tree_node_max = len(max(tree, default=0, key=len))
        node_len = max(tree_node_max, len(node))
        tree_len = len(tree)
        if node_len > node_max:
            node_max = node_len
        if tree_len > tree_max:
            tree_max = tree_len
    for node, tree in batch:
        pad_node(node, node_max)
        pad_tree(tree, tree_max, node_max)
        node_tensor = Tensor(node)
        tree_tensor = Tensor(node)
        new_batch.append((node_tensor, tree_tensor))
    return new_batch


class PadError(Exception):
    pass


class BaseIndexBuilder:
    def __init__(self, trees, max_indexes):
        self.trees = trees
        self.max_indexes = max_indexes

    def build_indexes(self):
        print("building_indexes...")
        indexes = []
        for tree_index in range(len(self.trees)):
            for tree_path_index in range(len(self.trees[tree_index].path)):
                indexes.append((tree_path_index, tree_index))
        random.shuffle(indexes)
        print('done with indexes. Length: ', min(len(indexes), self.max_indexes))
        return indexes[:self.max_indexes]


class PerTreeIndexBuilder(BaseIndexBuilder):
    def __init__(self, trees, max_indexes, indexes_per_tree):
        super().__init__(trees, max_indexes)
        self.indexes_per_tree = indexes_per_tree

    def build_indexes(self):
        print("building_indexes...")
        indexes = []
        for tree_index in range(len(self.trees)):
            path_indexes = range(len(self.trees[tree_index].path))
            sample = path_indexes if self.indexes_per_tree > len(path_indexes) else \
                random.sample(path_indexes, self.indexes_per_tree)
            for tree_path_index in sample:
                indexes.append((tree_path_index, tree_index))
        print('done with indexes. Length: ', min(len(indexes), self.max_indexes))
        random.shuffle(indexes)
        return indexes[:self.max_indexes]


class BaseSampleBuilder:
    def __init__(self, trees, node_tokenizer, tree_tokenizer):
        self.trees = trees
        self.node_tokenizer = node_tokenizer
        self.tree_tokenizer = tree_tokenizer
        self.node_max = 0
        self.tree_max = 0

    def build_samples(self, indexes):
        print("building samples...")
        samples = []
        i = 0
        for tree_index_path, tree_index in indexes:
            i += 1
            tokenized_node, tokenized_tree = self.build_sample(tree_index_path, tree_index)
            samples.append((tokenized_node, tokenized_tree))
        return samples

    def build_sample(self, tree_path_index, tree_index):
        tree: HtmlNode = self.trees[tree_index]
        node: HtmlNode = tree.path[tree_path_index]
        tokenized_node = self.node_tokenizer(node=node)
        node.mask_self()
        tokenized_tree = self.tree_tokenizer(tree=tree)
        node.unmask_self()
        tree_node_max = len(max(tokenized_tree, default=0, key=len))
        self.node_max = max(tree_node_max, self.node_max)
        self.tree_max = max(len(tokenized_tree), self.tree_max)
        return tokenized_node, tokenized_tree


class TransformerSampleBuilder(BaseSampleBuilder):
    def __init__(self, trees, node_tokenizer, tree_tokenizer):
        super().__init__(trees, node_tokenizer, tree_tokenizer)

    def build_sample(self, tree_path_index, tree_index):
        # print('~~~~~~~~~~~~~~ This is New ~~~~~~~~~~~')
        tree: HtmlNode = self.trees[tree_index]
        node: HtmlNode = tree.path[tree_path_index]
        tokenized_node = self.tree_tokenizer(tree=node)
        node.mask_affected()
        tokenized_tree = self.tree_tokenizer(tree=tree)
        node.unmask_affected()
        tree_node_max = len(max(tokenized_tree, default=0, key=len))
        node_node_max = len(max(tokenized_node, default=0, key=len))
        self.node_max = max(tree_node_max, node_node_max, self.node_max)
        self.tree_max = max(len(tokenized_tree), self.tree_max)
        return tokenized_node, tokenized_tree
