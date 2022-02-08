from argparse import Namespace, ArgumentParser
from typing import Tuple, List

from torch.utils.data import DataLoader
from project.current.dataloading import total_build_indexes, CustomTreeDataset
from project.current.frequency import build_trees, build_vocabs
from project.current.parsing import pickle_dump, pickle_load
from project.current.sparsing import random_sparse
from project.models.bow_implementation import bow_model

# Locations of relevant objects
file_loc = './data/'
models_loc = './project/models/'            # models


def reduce_trees(args: Namespace) -> None:
    if args.reduction == 'random':
        for tree in args.trees:
            random_sparse(tree=tree, goal_size=args.max_tree_size)
        pickle_dump(directory=args.setup_location + 'trees/trees_short', item=args.trees)
        args.reduction_function = random_sparse
    else:
        raise NoReduction


def set_index(args: Namespace) -> List[Tuple[int, int]]:
    if args.sampling == 'all':
        args.index_builder = total_build_indexes
        return total_build_indexes(trees=args.trees, size=args.indexes_size)
    else:
        raise NoIndex


def set_model(args: Namespace):
    if args.model_type == 'bow':
        args.model = bow_model
    else:
        raise NoModel


def main(args: Namespace) -> None:
    args.setup_location = file_loc + args.folder_name + '/'
    setup_location = args.setup_location
    if args.skip_setup:
        args.trees = pickle_load(directory=setup_location + 'trees/trees_short')
        args.tags = pickle_load(directory=setup_location + 'vocabs/tags')
        args.keys = pickle_load(directory=setup_location + 'vocabs/keys')
        args.values = pickle_load(directory=setup_location + 'vocabs/values')
        args.total = pickle_load(directory=setup_location + 'vocabs/total')
        train_dataloader = pickle_load(directory=setup_location + 'dataloaders/train')
        test_dataloader = pickle_load(directory=setup_location + 'dataloaders/test')
    else:
        if args.build_trees:
            build_trees(directory=setup_location, args=args)
        else:
            args.trees = pickle_load(directory=setup_location + 'trees/trees')

        if args.build_vocabs:
            build_vocabs(directory=setup_location, args=args)
        else:
            args.tags = pickle_load(directory=setup_location + 'vocabs/tags')
            args.keys = pickle_load(directory=setup_location + 'vocabs/keys')
            args.values = pickle_load(directory=setup_location + 'vocabs/values')
            args.total = pickle_load(directory=setup_location + 'vocabs/total')

        reduce_trees(args=args)
        for tree in args.trees:
            args.reduction_function(tree=tree, goal_size=args.max_tree_size)
            pickle_dump(directory=setup_location + 'trees/trees_short', item=args.trees)

        indexes = set_index(args=args)

        train_length = int(args.train_proportion * len(indexes))
        train_data = CustomTreeDataset(indexes=indexes[:train_length],
                                       tree_directory=setup_location + 'trees/trees_short', args=args)
        test_data = CustomTreeDataset(indexes=indexes[train_length:],
                                      tree_directory=setup_location + 'trees/trees_short', args=args)
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        pickle_dump(setup_location + 'dataloaders/train', train_dataloader)
        pickle_dump(setup_location + 'dataloaders/test', test_dataloader)

    set_model(args=args)
    args.model(args)
    pass

    # train_features, train_labels = next(iter(train_dataloader))
    # test_features, test_labels = next(iter(test_dataloader))


if __name__ == "__main__":
    parser = ArgumentParser(description='Process specifications')
    parser.add_argument('--pickle_trees', default=True)
    parser.add_argument('--folder_name', type=str, default='common_sites')
    parser.add_argument('--reduction', type=str, default='random')
    parser.add_argument('--build_vocabs', action='store_true')
    parser.add_argument('--include_data', action='store_true')
    parser.add_argument('--build_trees', action='store_true')
    parser.add_argument('--total_file_setup', action='store_true')
    parser.add_argument('--sampling', type=str, default='all')
    parser.add_argument('--indexes_size', type=int, default=200)
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--max_tree_size', type=int, default=500)
    parser.add_argument('--pad_value', type=int, default=0)  # pad_value currently equal to 'other' values
    parser.add_argument('--tag_other', type=int, default=0)
    parser.add_argument('--key_other', type=int, default=0)
    parser.add_argument('--value_other', type=int, default=0)
    parser.add_argument('--skip_setup', action='store_true')
    parser.add_argument('--model_type', type=str, default='bow')

    names: Namespace = parser.parse_args()
    main(names)


class NoIndex(Exception):
    pass


class NoReduction(Exception):
    pass


class NoModel(Exception):
    pass
