from argparse import Namespace
from typing import List

from project.current.parsing import HtmlNode
from project.current.frequency import  Vocabulary
import itertools
from torch import Tensor

build_vocab = 0
Node_Tokens = List[int]
Tree_Tokens = List[Node_Tokens]
Forest_Tokens = List[Tree_Tokens]
PAD_VALUE = 0


class BaseTokenizer:
    def __init__(self, vocabs: List[Vocabulary], total=False):
        self.offset = 0
        if total:
            self.set_total_vocab(vocabs[0])
            self.offset = len(vocabs[0])
        else:
            self.tags = vocabs[0]
            self.keys = vocabs[1]
            self.values = vocabs[2]


    def __call__(self, node: HtmlNode) -> List[int]:
        depth = self.handle_depth(node.depth)
        tag = self.handle_tag(node.tag)
        attrs = list(itertools.chain(*[self.handle_attr(key, value) for key, value in node.attrs]))
        data = self.handle_data(node.data)
        return depth + tag + attrs + data

    def handle_depth(self, depth: int) -> List[int]:
        return [depth + self.offset]

    def handle_tag(self, tag: str) -> List[int]:
        return [self.tags[tag]]

    def handle_attr(self, key: str, value: str) -> List[int]:
        return [self.keys[key], self.values[value]]

    def handle_data(self, data: str) -> List[int]:
        return []

    def set_total_vocab(self, total: Vocabulary) -> None:
        self.tags, self.keys, self.values = total, total, total
        self.offset = len(total)


class KeyOnlyTokenizer(BaseTokenizer):
    def handle_attr(self, key: str, value: int) -> List[int]:
        return [self.keys[key]]


class TreeTokenizer:
    def __init__(self, vocabs: List[Vocabulary], total=False, key_only=False):
        if key_only:
            self.node_tokenizer = KeyOnlyTokenizer(vocabs=vocabs, total=total)
        else:
            self.node_tokenizer = BaseTokenizer(vocabs=vocabs, total=total)

    def __call__(self, tree: HtmlNode) -> List[List[int]]:
        return [self.node_tokenizer(node) for node in tree.path]











def tokenize_node(node: HtmlNode, args: Namespace, length=None, total=False) -> Node_Tokens:
    # Return list [depth, tag_token, {key_token,value_token}*]
    if not total:
        tags, keys, values = args.tags, args.keys, args.values
        depth = [node.depth]
        tag = [tags[node.tag]]
        attrs = []
        add_on = []
        for key, value in node.attrs:
            attrs.append(keys[key])
            attrs.append(values[value])
    else:
        total_vocab = args.total
        depth = [node.depth + len(total_vocab)]
        tag = [total_vocab[node.tag]]
        attrs, add_on = [], []
        for key, value in node.attrs:
            attrs.append(total_vocab[key])
            attrs.append(total_vocab[value])
    if length is not None:  # for testing purposes
        add_on_length = length - len(tag) - len(attrs) - len(depth)
        add_on = [PAD_VALUE] * add_on_length
    return depth + tag + attrs + add_on


def tokenize_tree(tree: HtmlNode, args: Namespace, node_tokenizer: callable = tokenize_node, total=False) -> Tree_Tokens:
    return [node_tokenizer(node, args, total=total) for node in tree.path]


def tokenize_forest(trees: List[HtmlNode], args: Namespace, tree_tokenizer: callable = tokenize_tree) -> Forest_Tokens:  # return list of tokenized trees
    return [tree_tokenizer(tree, args) for tree in trees]


def node_token_to_node(node_tokens, args: Namespace) -> HtmlNode:
    tokens = [int(x) for x in node_tokens]
    depth = tokens[0]
    tag = tokens[1]
    attrs = []
    for i, val in enumerate(tokens):
        if not i % 2 and i >= 2:
            attr = (args.keys.reverse(tokens[i]), args.values.reverse(tokens[i + 1]))
            attrs.append(attr)
    return HtmlNode(args.tags.reverse(tag), attrs, depth=depth)


def tree_token_to_node(tree_tokens, args):
    path = [node_token_to_node(node_tokens, args) for node_tokens in tree_tokens]
    depths = [node.depth for node in path]
    print([node.tag for node in path])
    print(depths)
    for child_index, child_node in enumerate(path):
        for depth_index, depth in enumerate(depths):
            if depth_index >= child_index + 1 and depth == child_node.depth - 1:
                # print('indexes: ', depth_index, depth)
                # print('child_index: ', child_index)
                path[depth_index].children.append(child_node)
                break
    root = path[-1]
    root.build_path()
    return root

