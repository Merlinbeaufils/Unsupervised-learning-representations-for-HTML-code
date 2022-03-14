import itertools
from argparse import Namespace
from typing import List

from torch import Tensor

from project.frequency import Vocabulary
from project.parsing import HtmlNode

build_vocab = 0
Node_Tokens = List[int]
Tree_Tokens = List[Node_Tokens]
Forest_Tokens = List[Tree_Tokens]
PAD_VALUE = 0


class BaseTokenizer:
    def __init__(self, vocabs: List[Vocabulary], total=False):
        """
        Class to tokenize trees into nested lists of ints
        :param vocabs: vocab for tokenization
        :param total: if True, use one joint vocab, not separate vocabs
        """
        self.offset = 0
        if total:
            total_vocab = vocabs[0]
            self.tags, self.keys, self.values = total_vocab, total_vocab, total_vocab
            self.offset = len(total_vocab)
        else:
            self.tags = vocabs[0]
            self.keys = vocabs[1]
            self.values = vocabs[2]
        self.vocabs = vocabs
        self.total = total

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

    def back_to_node(self, node_token: Tensor) -> HtmlNode:
        """ Build node back from token
        Dont use"""
        tag_vocab, key_vocab, value_vocab = self.vocabs * 3 if self.total else self.vocabs
        attrs = []
        for i, val in enumerate(node_token):
            val = int(val)
            if i == 0:
                depth = node_token[0]
            elif i == 1:
                tag = tag_vocab.reverse(val)
            elif i % 2 == 0:
                attrs.append((key_vocab.reverse(val), value_vocab.reverse(node_token[i + 1])))
        return HtmlNode(depth=depth, tag=tag, attrs=attrs)

    def back_to_tree(self, tree_token) -> HtmlNode:
        """ Build tree from token
        Dont use"""
        path = [self.back_to_node(node_token) for node_token in tree_token]
        depths = [node.depth for node in path]
        # print([node.tag for node in path])
        # print(depths)
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


class TransformerTreeTokenizer(BaseTokenizer):
    def __init__(self, total_vocab):
        """
        Builds 1d token with end-of-node and start-of-node tags.
        :param total_vocab: joint vocab
        """
        super().__init__([total_vocab], total=True)
        self.IGNORE_IDX = total_vocab["<ignore>"]
        self.MASK_IDX = total_vocab["<mask>"]
        self.OOV_IDX = total_vocab["<oov>"]
        self.EON_IDX = total_vocab["<eon>"]
        self.SON_IDX = total_vocab["<son>"]

    def __call__(self, tree: HtmlNode):
        feature, label = [], []
        for node in tree.path:
            node_f, node_l = self.handle_node(node)
            feature += node_f
            label += node_l
        assert len(feature) == len(label)
        return feature, label

    def handle_node(self, node):
        node_token = [self.SON_IDX] + super().__call__(node) + [self.EON_IDX]
        if node.mask_val:
            label = node_token
            feature = [self.MASK_IDX] * len(label)
        else:
            feature = node_token
            label = [self.IGNORE_IDX] * len(feature)
        return feature, label


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

