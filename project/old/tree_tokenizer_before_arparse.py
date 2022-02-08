from typing import List
from code.deprecated.frequency import *
build_vocab = 0
Node_Tokens = List[int]
Tree_Tokens = List[Node_Tokens]
Forest_Tokens = List[Tree_Tokens]
PAD_VALUE = 0


def pickle_load(directory: str):
    with open(directory, 'rb') as handle:
        return pickle.load(handle)


# possible values for attributes_setup: keyvalue, string
attributes_setup = "keyvalue"
if build_vocab == 1:
    build_vocabs(tag_floor=0, key_floor=0, value_floor=3, attr_floor=0)
attributes: Vocabulary = pickle_load('./analysis/vocabularies/attr_pickle')
keys:       Vocabulary = pickle_load('./analysis/vocabularies/key_pickle')
tags:       Vocabulary = pickle_load('./analysis/vocabularies/tag_pickle')
values:     Vocabulary = pickle_load('./analysis/vocabularies/value_pickle')
tags['mask'] = len(tags)


def tokenize_node(node: HtmlNode, length=None) -> Node_Tokens:
    # Return list of tag token followed by however many key, value token pairs
    tag = [tags[node.tag]]
    attrs = []
    add_on = []
    for attr in node.attrs:
        key, value = attr
        attrs.append(keys[key])
        attrs.append(values[value])
    if length is not None:
        add_on_length = length - len(tag) - len(attrs)
        add_on = [PAD_VALUE] * add_on_length
    return tag + attrs + add_on


def tokenize_tree(tree: HtmlNode) -> Tree_Tokens:  # return list of tokenized nodes
    return [tokenize_node(node) for node in tree.path]


def tokenize_forest(trees: List[HtmlNode]) -> Forest_Tokens:  # return list of tokenized trees
    return [tokenize_tree(tree) for tree in trees]


def token_to_node(tokens: List[int]) -> HtmlNode:
    tag = tokens[0]
    attrs = []
    for i, val in enumerate(tokens):
        if i % 2:
            attr = (keys.reverse(tokens[i]), values.reverse(tokens[i + 1]))
            attrs.append(attr)
    return HtmlNode(tags.reverse(tag), attrs)
