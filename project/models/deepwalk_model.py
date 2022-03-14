import random
from typing import List, Tuple

import torch
import torch.nn as nn

from project.dataloading import BaseTreeDataset
from project.parsing import HtmlNode


def random_walk(tree: HtmlNode, length, walk=None):
    walk = [] if walk is None else walk

    next_node = tree.children[random.randint(0, len(tree.children))] \
        if tree.children else tree.root_node

    return walk + [next_node] + random_walk(next_node, length - 1)


class DeepWalkDataset(BaseTreeDataset):
    def __init__(self, trees, vocabs, indexes_length, total=True, key_only=False, build_samples=True):
        super().__init__(trees, vocabs, indexes_length, total, key_only, build_samples)

