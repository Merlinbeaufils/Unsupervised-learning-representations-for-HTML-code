from typing import List
from frequency import *
build_vocab = 0
Node_Tokens = List[int]
Tree_Tokens = List[Node_Tokens]
Forest_Tokens = List[Tree_Tokens]


def pickle_load(directory: str):
    with open(directory, 'rb') as handle:
        return pickle.load(handle)


# possible values for attributes_setup: keyvalue, string
attributes_setup = "keyvalue"
if build_vocab == 1:
    build_vocabs(tag_floor=0, key_floor=0, value_floor=0, attr_floor=0)
attributes: Vocabulary = pickle_load('./analysis/vocab/attr_pickle')
keys:       Vocabulary = pickle_load('./analysis/vocab/key_pickle')
tags:       Vocabulary = pickle_load('./analysis/vocab/tag_pickle')
values:     Vocabulary = pickle_load('./analysis/vocab/value_pickle')
tags['mask'] = len(tags)


def tokenize_node(node: HtmlNode) -> Node_Tokens:
    # Return list of tag token followed by however many key, value token pairs
    tag = [tags[node.tag]]
    attrs = []
    for attr in node.attrs:
        key, value = attr
        attrs.append(keys[key])
        attrs.append(values[value])
    return tag + attrs


def tokenize_tree(tree: HtmlNode) -> Tree_Tokens:  # return list of tokenized nodes
    return [tokenize_node(node) for node in tree.path]


def tokenize_forest(trees: List[HtmlNode]) -> Forest_Tokens:  # return list of tokenized trees
    return [tokenize_tree(tree) for tree in trees]
