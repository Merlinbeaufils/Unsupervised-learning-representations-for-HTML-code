import os
from argparse import Namespace, ArgumentParser
from typing import Tuple, List

import pandas
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, random_split

from project.dataloading import BaseTreeDataset, ContTreeDataset, TransformerTreeDataset, TreeClassifierDataset
from project.finetune_model_pipe import TreeClassifier
from project.frequency import build_trees, build_vocabularies, build_files, Vocabulary, build_trees_and_files, \
    term_frequency
from project.pretraining_model_pipe import BaseModel
from project.models.transformer_implementation import TransformerModule
from project.parsing import pickle_dump, pickle_load, HtmlNode
from project.sparsing import random_sparse, sparse_depth, length_depth_reduction
from project.tree_tokenizer import BaseTokenizer


class NoIndex(Exception):
    pass


class NoReduction(Exception):
    pass


class NoModel(Exception):
    pass


class NoDataloader(Exception):
    pass


class Stop(Exception):
    pass


torch.manual_seed(1)

file_loc = './data/'


def reduce_trees(reduction: str, trees: List[HtmlNode], args: Namespace) -> None:
    print('reducing trees ... ')
    if reduction == 'random':
        function = random_sparse
        kwargs = {'goal_size': args.max_tree_size}
    elif reduction == 'depth':
        function = sparse_depth
        kwargs = {'depth': args.max_depth}
    elif reduction == 'both':
        function = length_depth_reduction
        kwargs = {'goal_size': args.goal_size, 'depth': args.max_depth}
    else:
        raise NoReduction
    for tree in trees:
        function(tree, **kwargs)
    pickle_dump(directory=args.setup_location + 'trees/trees_short', item=trees)
    args.reduction_function = function


def set_model(args, dataset):
    print('building model ...')
    config = args.configuration.lower()
    if config in ['bow', 'lstm', 'transformer']:
        if config == 'bow':
            node_model, tree_model = ('bow', None)
        elif config == 'lstm':
            node_model, tree_model = ('lstm', None) if not args.separate else ('lstm', 'lstm')
        elif config == 'transformer':
            node_model, tree_model = ('transformer', None) if not args.separate else ('transformer', 'transformer')
        kwargs = {'dataset': dataset, 'node_model_type': node_model,
                  'vocab_size': len(args.total), 'tree_model_type': tree_model,
                  'optimizer_type': args.optimizer, 'batch_size': args.batch_size,
                  'lr': args.lr, 'loss_type': args.loss, 'similarity_type': args.similarity,
                  'embedding_dim': args.embedding_dim, 'train_proportion': args.train_proportion,
                  'num_cpus': args.num_cpus}
        model = BaseModel(**kwargs)
        args.log_name = config
    elif config == 'transformer.':
        kwargs = {'n_code': args.n_code, 'n_heads': args.n_heads, 'embed_size': args.embedding_dim,
                  'inner_ff_size': args.embedding_dim * 4,
                  'n_embeddings': len(dataset.vocab) + args.max_depth,
                  'max_seq_len': args.max_seq_len, 'dropout': args.dropout}

        optim_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': (.9, .999)}

        loader_kwargs = {'num_workers': args.num_cpus, 'shuffle': args.shuffle, 'drop_last': args.drop_last,
                         'pin_memory': args.pin_memory, 'batch_size': args.batch_size}
        model = TransformerModule(dataset, kwargs=kwargs, optim_kwargs=optim_kwargs, loader_kwargs=loader_kwargs)
        args.log_name = 'transformer'

    else:
        raise NoModel
    return model


def set_dataloader(dataloader: str, trees: List[HtmlNode], indexes_size: int,
                   train_proportion: int, args: Namespace) -> Tuple[DataLoader, DataLoader, BaseTreeDataset]:
    print('setting datasets...')
    vocabs = [args.total] if args.total_vocab else [args.tags, args.keys, args.values]
    vocab = args.total
    data_config = 'keys_only' if args.keys_only else 'normal'
    data_config = 'no_keys' if args.no_keys else data_config
    if dataloader == 'base':
        dataset = BaseTreeDataset(trees=trees, indexes_length=indexes_size,
                                  total=True, key_only=True, vocabs=vocabs, index_config=args.index_config,
                                  per_tree=args.indexes_per_tree, sample_config=args.sample_config, no_keys=args.no_keys)


    elif dataloader == 'Cont':
        dataset = ContTreeDataset(trees=trees, indexes_length=indexes_size,
                                  total=True, key_only=True, vocabs=vocabs)

    elif dataloader == 'transformer':
        dataset = TransformerTreeDataset(trees=trees, total_vocab=vocab, indexes_length=indexes_size,
                                         key_only=False, max_seq_len=args.max_seq_len, index_config=args.index_config,
                                         per_tree=args.per_tree)
    else:
        raise NoDataloader
    train_size = int(train_proportion * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    # train_dataloader, test_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True), \
    #                                    DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    os.makedirs(args.setup_location + 'dataloaders', mode=0o777, exist_ok=True)
    # pickle_dump(args.setup_location + 'dataloaders/train_' + dataloader, train_dataloader)
    # pickle_dump(args.setup_location + 'dataloaders/test_' + dataloader, test_dataloader)
    pickle_dump(args.setup_location + 'dataloaders/dataset_' + data_config, dataset)
    return None, None, dataset


class NoFramework(Exception):
    pass


def main(args: Namespace) -> None:
    args.setup_location = file_loc + args.folder_name + '/'
    setup_location = args.setup_location
    if args.framework == 'pretrain':
        if args.skip_setup:
            args.trees = pickle_load(directory=setup_location + 'trees/trees_short')
            args.tags = pickle_load(directory=setup_location + 'vocabs/tags')
            args.keys = pickle_load(directory=setup_location + 'vocabs/keys')
            args.values = pickle_load(directory=setup_location + 'vocabs/values')
            args.total = pickle_load(directory=setup_location + 'vocabs/total')
            # train_dataloader = pickle_load(directory=setup_location + 'dataloaders/train_' + args.dataloader)
            # test_dataloader = pickle_load(directory=setup_location + 'dataloaders/test_' + args.dataloader)
            data_config = 'keys_only' if args.keys_only else 'normal'
            data_config = 'no_keys' if args.no_keys else data_config

            dataset = pickle_load(directory=setup_location + 'dataloaders/dataset_' + data_config)
        else:
            print('building trees and files...')
            args.trees = build_trees_and_files(directory=args.setup_location, pandas=not args.not_pandas, max_trees=args.num_trees)
            # if args.build_trees:
            #     args.trees = build_trees(directory=setup_location, pickle_trees=args.pickle_trees, pandas=args.pandas)
            # else:
            #     args.trees = pickle_load(directory=setup_location + 'trees/trees')

            print('building vocabs ...')
            # build_files(setup_location, setup_location + 'text_files', key_only=args.key_only, pandas=args.pandas)
            args.tags, args.keys, args.values, args.total = \
                build_vocabularies(directory=setup_location, total_floor=args.total_floor)

            args.data_matrix, args.vectorizer = term_frequency(setup_location + 'text_files/data.txt')

            # else:
            #     args.tags = pickle_load(directory=setup_location + 'vocabs/tags')
            #     args.keys = pickle_load(directory=setup_location + 'vocabs/keys')
            #     args.values = pickle_load(directory=setup_location + 'vocabs/values')
            #     args.total = pickle_load(directory=setup_location + 'vocabs/total')

            reduce_trees(args.reduction, args.trees, args)

            train_dataloader, test_dataloader, dataset = set_dataloader(dataloader=args.dataloader, trees=args.trees,
                                                                        indexes_size=args.indexes_size,
                                                                        train_proportion=args.train_proportion, args=args)
            print(len(dataset))
            if args.stop:
                raise Stop

        # train_features, train_labels = next(iter(train_dataloader))
        # test_features, test_labels = next(iter(train_dataloader))
        # feature, label = train_features[0], train_labels[0]

        model = set_model(args, dataset)

    elif args.framework == 'finetune':
        args.total = pickle_load(directory=setup_location + 'vocabs/total')
        print('building dataset...')
        if args.skip_setup:
            dataset = pickle_load(directory=setup_location + 'dataloaders/dataset_' + 'classifier')
        else:
            dataset = TreeClassifierDataset(args.setup_location + 'websites.feather', args.indexes_size, [args.total],
                                            True, key_only=args.key_only, no_keys=args.no_keys)
            if args.indexes_size < len(dataset):
                dataset, _ = random_split(dataset, [args.indexes_size, len(dataset)-args.indexes_size])

            pickle_dump(args.setup_location + 'dataloaders/dataset_classifier', dataset)
        print('building model...')
        model = TreeClassifier('res_and_ckpts/' + args.experiment_name + '/checkpoints/pretrain/' + args.configuration + '.ckpt',
                               dataset, args.embedding_dim, dataset.num_labels, 'sgd',
                               args.train_proportion, args.num_cpus, args.configuration, args.batch_size, args.lr)
        args.log_name = 'bow'

    else:
        raise NoFramework
    save_folder = 'res_and_ckpts/' + args.experiment_name + '/tb_logs/' + args.framework
    os.makedirs(save_folder, mode=0o777, exist_ok=True)
    os.makedirs('res_and_ckpts/' + args.experiment_name + '/checkpoints/' + args.framework, mode = 0o777, exist_ok=True)
    logger = TensorBoardLogger(save_folder, name=args.log_name)

    if args.num_gpus > 0:
        model = model.cuda()
    trainer = Trainer(
        gpus=args.num_gpus,
        logger=[logger],
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        default_root_dir='res_and_ckpts/' + args.experiment_name + '/checkpoints/' + args.framework
    )
    trainer.fit(model)
    torch.save(model, 'res_and_ckpts/' + args.experiment_name + '/checkpoints/' + args.framework + '/' + args.configuration + '.ckpt')
    pass


def test_some_stuff():
    total_vocab: Vocabulary = pickle_load('./data/common_sites/vocabs/total')
    # train_dataloader = pickle_load('./data/common_sites/dataloaders/train_base')
    trees = pickle_load('./data/common_sites/trees/trees')
    indexes_size = 100
    vocabs = [total_vocab]
    dataset = BaseTreeDataset(trees=trees, indexes_length=indexes_size,
                              total=True, key_only=True, vocabs=vocabs)
    # dataset = pickle_load('./data/common_sites/dataloaders/dataset_base')
    train_dataloader = DataLoader(dataset, 1, True)
    features, labels = next(iter(train_dataloader))
    feature, label = features[0], labels[0]
    tokenizer = BaseTokenizer(vocabs=[total_vocab], total=True)
    node = tokenizer.back_to_node(feature)
    tree = tokenizer.back_to_tree(label)
    print('hi')


if __name__ == "__main__":
    parser = ArgumentParser(description='Process specifications')
    parser.add_argument('--pickle_trees', action='store_true')
    parser.add_argument('--folder_name', type=str, default='feather')
    parser.add_argument('--reduction', type=str, default='random')
    parser.add_argument('--build_vocabs', action='store_true')
    parser.add_argument('--include_data', action='store_true')
    parser.add_argument('--build_trees', action='store_true')
    parser.add_argument('--total_file_setup', action='store_true')
    parser.add_argument('--sampling', type=str, default='all')
    parser.add_argument('--indexes_size', type=int, default=10_000)
    parser.add_argument('--train_length', type=int, default=400)
    parser.add_argument('--test_length', type=int, default=100)
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--max_tree_size', type=int, default=500)
    parser.add_argument('--pad_value', type=int, default=0)  # pad_value currently equal to 'other' values
    parser.add_argument('--tag_other', type=int, default=0)
    parser.add_argument('--key_other', type=int, default=0)
    parser.add_argument('--value_other', type=int, default=0)
    parser.add_argument('--skip_setup', action='store_true')
    parser.add_argument('--tree_model_type', type=str, default=None)
    parser.add_argument('--node_model_type', type=str, default='flat')
    parser.add_argument('--dataloader', type=str, default='base')
    parser.add_argument('--total_floor', type=int, default=2)
    parser.add_argument('--key_only', action='store_true')
    parser.add_argument('--total_vocab', action='store_true')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--goal_size', type=int, default=500)
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--num_cpus', type=int, default=2)
    parser.add_argument('--configuration', type=str, default='bow')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--n_code', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--droupout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--not_pandas', action='store_true')
    parser.add_argument('--num_trees', type=int, default=200)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--indexes_per_tree', type=int, default=10)
    parser.add_argument('--index_config', type=str, default='per_tree')
    parser.add_argument('--sample_config', type=str, default='base')
    parser.add_argument('--framework', type=str, default='pretrain')
    parser.add_argument('--experiment_name', type=str, default='base')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--no_keys', action='store_true')

    names: Namespace = parser.parse_args()
    names.total_vocab = True  # change this at some point
    if names.test:
        test_some_stuff()
    main(names)
