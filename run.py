from bow_implementation import *


def set_reduction(args: Namespace) -> None:
    if args.reduction == 'random':
        args.reduction_function = random_sparse
    else:
        print('no_reduction_built')


def set_index_builder(args: Namespace) -> None:
    if args.sampling == 'all':
        args.index_builder = total_build_indexes
    else:
        print('no indexing built')


def set_model(args: Namespace):
    if args.model_type == 'bow':
        args.model = bow_model


def main(args: Namespace) -> None:
    if args.skip_setup:
        args.trees = pickle_load(directory='./tree_directory_short')
        args.tags: Vocabulary = pickle_load(directory='./vocab/tags')
        args.keys: Vocabulary = pickle_load(directory='./vocab/keys')
        args.values: Vocabulary = pickle_load(directory='./vocab/values')
        train_dataloader = pickle_load(directory='./dataloaders/train_' + args.file_directory[2:])
        test_dataloader = pickle_load(directory='./dataloaders/test_' + args.file_directory[2:])
    else:
        if args.build_trees or args.total_file_setup:
            build_trees(directory=args.file_directory, args=args)
        else:
            args.trees = pickle_load(directory='./tree_directory')

        if args.build_vocabs or args.total_file_setup:
            build_vocabs(directory=args.file_directory, args=args)
        else:
            args.tags: Vocabulary = pickle_load(directory='./vocab/tags')
            args.keys: Vocabulary = pickle_load(directory='./vocab/keys')
            args.values: Vocabulary = pickle_load(directory='./vocab/values')

        set_reduction(args=args)
        for tree in args.trees:
            args.reduction_function(tree=tree, goal_size=args.max_tree_size)
            pickle_dump(directory=args.file_directory + '_short', item=args.trees)

        set_index_builder(args=args)
        indexes = args.index_builder(size=args.indexes_size)

        train_length = int(args.train_proportion * len(indexes))
        train_data = CustomTreeDataset(indexes=indexes[:train_length], tree_directory='./tree_directory_short', args=args)
        test_data = CustomTreeDataset(indexes=indexes[train_length:], tree_directory='./tree_directory_short', args=args)
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        pickle_dump('./dataloaders/train_' + args.file_directory[2:], train_dataloader)
        pickle_dump('./dataloaders/test_' + args.file_directory[2:], test_dataloader)

    set_model(args=args)
    args.model(args)
    pass

    train_features, train_labels = next(iter(train_dataloader))
    test_features, test_labels = next(iter(test_dataloader))
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description='Process specifications')
    parser.add_argument('--pickle_trees', default=True)
    parser.add_argument('--file_directory', type=str, default='./common_sites')
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
    pass
