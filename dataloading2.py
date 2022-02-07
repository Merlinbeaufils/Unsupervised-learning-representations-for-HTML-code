# from typing import List, Tuple
from tree_tokenizer2 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
import random
from sparsing import random_sparse

Tuple_list = List[Tuple[int, int]]
Sample = Tuple[Node_Tokens, Tree_Tokens]
Samples = List[Sample]
Forest = List[HtmlNode]
Tensorized_Sample = Tuple[Tensor, Tensor]
# PAD_VALUE = 0
SIZE = 200
MAX_SIZE = 500
MY_COLLATE = 0
upgrade_collate = 0


def total_build_indexes(size: int = 2000) -> Tuple_list:  # Builds extensive indexes of every tree-subtree pair
    indexes = []
    trees = pickle_load('./tree_directory')
    for tree in trees:
        random_sparse(tree, MAX_SIZE)
    with open('./tree_directory_short', 'wb') as handle:
        pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for tree_index in range(len(trees)):
        for tree_path_index in range(len(trees[tree_index].path)):
            indexes.append((tree_path_index, tree_index))

    random.shuffle(indexes)
    print('done with indexes. Length: ', len(indexes))
    return indexes[:size]


class CustomTreeDataset(Dataset):  # Tree dataset class allowing handling of html trees
    def __init__(self, indexes: Tuple_list, tree_directory: str, args: Namespace):
        # indexes is a list of (tree_path_index, tree_index) tuples indicating which node and
        super().__init__()
        self.indexes = indexes
        self.trees:   Forest = pickle_load(tree_directory)
        self.samples: Samples = []
        self.tree_max: int = 0
        self.node_max: int = 0
        self.my_collate = 0
        self.build_samples(indexes, args)
        if not self.my_collate:
            self.padding_tokens()
        if upgrade_collate:
            self.upgrade_collate()

    def __getitem__(self, index: int) -> Tensorized_Sample:  # returns a (subtree, tree) pair. Tokenized
        token_node, token_tree = self.samples[index]
        return Tensor(token_node), Tensor(token_tree)

    def __len__(self) -> int:  # returns # of samples
        return len(self.indexes)

    def build_samples(self, indexes: Tuple_list, args: Namespace) -> None:  # Creates a list of (tokenized subtree, tokenized tree) pairs
        self.samples.clear()
        i = 0
        for tree_index_path, tree_index in indexes:
            i += 1
            node: HtmlNode = self.trees[tree_index].path[tree_index_path]
            tree: HtmlNode = self.trees[tree_index]
            tokenized_node = tokenize_node(node, args)
            node.mask_self()
            tokenized_tree = tokenize_tree(tree, args)
            node.unmask_self()
            tree_node_max = len(max(tokenized_tree, default=0, key=len))
            if not self.my_collate:
                node_len = max(tree_node_max, len(tokenized_node))
                if node_len > self.node_max:
                    self.node_max = node_len
                tree_len = len(tokenized_tree)
                if tree_len > self.tree_max:
                    self.tree_max = tree_len
            self.samples.append((tokenized_node, tokenized_tree))

    def padding_tokens(self) -> None:
        for node_sample, tree_sample in self.samples:
            pad_node(node_sample, self.node_max)
            pad_tree(tree_sample, self.tree_max, self.node_max)

    def reduce_trees(self, max_size=MAX_SIZE):
        for tree in self.trees:
            random_sparse(tree, max_size)


class MlmTreeDataset(CustomTreeDataset):
    def __init__(self, tokenizer):
        super().__init__()
        self.node_tokenizer = tokenizer

    def build_samples(self, indexes: Tuple_list, args: Namespace):
        self.samples.clear()
        i = 0
        for tree_index_path, tree_index in indexes:
            i += 1
            node: HtmlNode = self.trees[tree_index].path[tree_index_path]
            tree: HtmlNode = self.trees[tree_index]
            tokenized_node = self.tokenizer(node, args)
            node.mask_self()
            tokenized_tree = tokenize_tree(tree, args, self.tokenizer)
            node.unmask_self()
            tree_node_max = len(max(tokenized_tree, default=0, key=len))
            if not self.my_collate:
                node_len = max(tree_node_max, len(tokenized_node))
                if node_len > self.node_max:
                    self.node_max = node_len
                tree_len = len(tokenized_tree)
                if tree_len > self.tree_max:
                    self.tree_max = tree_len
            self.samples.append((tokenized_node, tokenized_tree))


def pad_tree(tree: list, length_tree: int, length_node: int) -> None:
    for node in tree:
        pad_node(node, length_node)
    while len(tree) < length_tree:
        tree.append([PAD_VALUE] * length_node)
    if len(tree) > length_tree:
        print('fuck')


def pad_node(node: list, length_node: int) -> None:
    while len(node) < length_node:
        node.append(PAD_VALUE)
    if len(node) > length_node:
        print('fuck')


# dataset = CustomTreeDataset(indexes, './tree_directory')
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


def basic_data_loader_build(args: Namespace, size: int = SIZE) -> (DataLoader, DataLoader):
    indexes = total_build_indexes(size)
    train_length = int(0.8 * len(indexes))
    # test_length = len(indexes) - train_length
    training_data = CustomTreeDataset(indexes[:train_length], './tree_directory_short', args)
    test_data = CustomTreeDataset(indexes[train_length:], './tree_directory_short', args)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader

# train_dataloader_, test_dataloader_ = basic_data_loader_build()
# train_features_, train_labels_ = next(iter(train_dataloader_))
# pass