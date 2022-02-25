import random

from project.parsing import HtmlNode


def random_sparse(tree: HtmlNode, goal_size): # reduce number of nodes in tree to at most goal_size
    """
    Randomnly drop nodes from tree until goal size is reached.
    :param tree: tree to be reduced
    :param goal_size: max size of tree
    :return: reduced tree
    """
    while len(tree.path) > goal_size:
        target_node = tree.path[random.randint(0, len(tree.path) - 2)]
        target_node.father.children = [child for child in target_node.father.children if child != target_node]
        tree.build_path()
    return tree


def sparse_depth(tree: HtmlNode, depth: int = 5):
    """
    removes all nodes with higher depth than depth
    :param tree: tree to be reduced
    :param depth: max_depth
    :return: reduced tree
    """
    for node in tree.path:
        if node.depth > depth:
            node.father.children = [child for child in node.father.children if child != node]
    tree.build_path()
    return tree


def length_depth_reduction(tree: HtmlNode, goal_size, depth):
    """ Applies both random and depth reductions """
    random_sparse(tree, goal_size)
    sparse_depth(tree, depth)
    return tree
