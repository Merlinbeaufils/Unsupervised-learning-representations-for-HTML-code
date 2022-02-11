# from typing import List, Tuple
import random
from typing import Tuple, List
from argparse import Namespace

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from project.current.parsing import pickle_load, pickle_dump
from project.current.sparsing import random_sparse
from project.current.frequency import Vocabulary
from project.current.tree_tokenizer import Node_Tokens, Tree_Tokens, BaseTokenizer, KeyOnlyTokenizer,TreeTokenizer
from project.current.parsing import HtmlNode


Sample = Tuple[Node_Tokens, Tree_Tokens]
Samples = List[Sample]
TensorizedSample = Tuple[Tensor, Tensor]
Forest = List[HtmlNode]
PAD_VALUE = 0


class BaseTreeDataset(Dataset):  # Tree dataset class allowing handling of html trees
    def __init__(self, trees: List[HtmlNode], vocabs: List[Vocabulary],
                 indexes_length=1000, total: bool = False, key_only: bool = False):
        # indexes is a list of (tree_path_index, tree_index) tuples indicating (node, tree)
        super().__init__()
        self.trees:   Forest = trees
        self.samples: List[Tuple[Node_Tokens, Tree_Tokens]] = []
        self.tree_max: int = 0
        self.node_max: int = 0
        self.total = total
        self.key_only = key_only
        self.node_tokenizer, self.tree_tokenizer = self.set_tokenizers(vocabs=vocabs)
        self.indexes = self.build_indexes(indexes_length)
        self.build_samples(self.indexes)
        self.padding_tokens()

    def __getitem__(self, index: int) -> TensorizedSample:  # returns a (subtree, tree) pair. Tokenized
        token_node, token_tree = self.samples[index]
        return Tensor(token_node), Tensor(token_tree)

    def __len__(self) -> int:  # returns # of samples
        return len(self.indexes)

    def build_indexes(self, indexes_length) -> List[Tuple[int, int]]:
        indexes = []
        for tree_index in range(len(self.trees)):
            for tree_path_index in range(len(self.trees[tree_index].path)):
                self.handle_index(indexes, tree_path_index, tree_index)
                if len(indexes) == indexes_length:
                    random.shuffle(indexes)
                    print('done with indexes. Length: ', len(indexes))
                    return indexes
        random.shuffle(indexes)
        print('done with indexes. Length: ', len(indexes))
        return indexes

    def handle_index(self, indexes, tree_path_index, tree_index):
        indexes.append((tree_path_index, tree_index))

    def build_samples(self, indexes):
        self.samples.clear()
        i = 0
        for tree_index_path, tree_index in indexes:
            i += 1
            tokenized_node, tokenized_tree = self.build_sample(tree_index_path, tree_index)
            self.samples.append((tokenized_node, tokenized_tree))

    def build_sample(self, tree_index_path, tree_index) -> Tuple[List, List]:
        node: HtmlNode = self.trees[tree_index].path[tree_index_path]
        tree: HtmlNode = self.trees[tree_index]
        tokenized_node = self.node_tokenizer(node=node)
        node.mask_self()
        tokenized_tree = self.tree_tokenizer(tree=tree)
        node.unmask_self()
        tree_node_max = len(max(tokenized_tree, default=0, key=len))
        self.node_max = max(tree_node_max, self.node_max)
        self.tree_max = max(len(tokenized_tree), self.tree_max)
        return tokenized_node, tokenized_tree

    def padding_tokens(self) -> None:
        for node_sample, tree_sample in self.samples:
            pad_node(node_sample, self.node_max)
            pad_tree(tree_sample, self.tree_max, self.node_max)

    def reduce_trees(self, max_size=500):
        for tree in self.trees:
            random_sparse(tree, max_size)

    def set_tokenizers(self, vocabs: List[Vocabulary]):
        if self.key_only:
            node_tokenizer = KeyOnlyTokenizer(vocabs=vocabs, total=self.total)
        else:
            node_tokenizer = BaseTokenizer(vocabs=vocabs, total=self.total)
        tree_tokenizer = TreeTokenizer(vocabs=vocabs, total=self.total)
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
