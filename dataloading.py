from parsing import *
from frequency import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random


def pickle_load(directory):
    with open(directory, 'rb') as handle:
        return pickle.load(handle)


def total_build_indexes():
    indexes = []
    trees = pickle_load('./tree_directory')
    for tree_index in range(len(trees)):
        for tree_path_index in range(len(trees[tree_index].path)):
            indexes.append((tree_path_index, tree_index))
    return indexes


class CustomTreeDataset(Dataset):
    def __init__(self, indexes, tree_directory):
        # indexes is a list of (tree_path_index, tree_index) tuples indicating which node and
        super().__init__()
        self.indexes = indexes
        self.trees = pickle_load(tree_directory)
        self.samples = []
        self.build_samples(indexes)

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.indexes)

    def build_samples(self, indexes):
        self.samples = []
        for tree_index_path, tree_index in indexes:
            node = self.trees[tree_index].path[tree_index_path]
            tree = self.trees[tree_index].mask(tree_index_path)
            self.samples.append((node, tree))
# dataset = CustomTreeDataset(indexes, './tree_directory')

def basic_data_loader_build():
    indexes = total_build_indexes()
    random.shuffle(indexes)
    train_length = int(0.8 * len(indexes))
    test_length = len(indexes) - train_length
    training_data = CustomTreeDataset(indexes[:train_length], './tree_directory')
    test_data = CustomTreeDataset(indexes[test_length:], './tree_directory')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader

basic_data_loader_build()
