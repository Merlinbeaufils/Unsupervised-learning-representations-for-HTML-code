from typing import List
from frequency2 import *
build_vocab = 0
Node_Tokens = List[int]
Tree_Tokens = List[Node_Tokens]
Forest_Tokens = List[Tree_Tokens]
PAD_VALUE = 0




def tokenize_node(node: HtmlNode, args: Namespace, length=None) -> Node_Tokens:
    # Return list [depth, tag_token, {key_token,value_token}*]
    tags, keys, values = args.tags, args.keys, args.values
    depth = [node.depth]
    tag = [tags[node.tag]]
    attrs = []
    add_on = []
    for attr in node.attrs:
        key, value = attr
        attrs.append(keys[key])
        attrs.append(values[value])
    if length is not None:  # for testing purposes
        add_on_length = length - len(tag) - len(attrs) - len(depth)
        add_on = [PAD_VALUE] * add_on_length
    return depth + tag + attrs + add_on


def tokenize_tree(tree: HtmlNode, args: Namespace, node_tokenizer: callable = tokenize_node) -> Tree_Tokens:
    return [node_tokenizer(node, args) for node in tree.path]


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
