from project.parsing import pickle_load


def function():
    trees = pickle_load('./data/common_sites/trees/trees')
    max_depth = max([max([node.depth for node in tree.path]) for tree in trees])
    pass

function()
