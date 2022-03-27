from typing import List, Tuple

import plotly
import numpy as np
import plotly.graph_objs as go
import torch
from pyarrow import feather
from sklearn.decomposition import PCA
import pandas
from torch import Tensor

from project.frequency import Vocabulary
from project.parsing import pickle_load, strings_to_trees, string_to_tree, pickle_dump
from project.tree_tokenizer import TreeTokenizer
from project.sparsing import length_depth_reduction
from project.dataloading import pad_tree


def display_pca_scatterplot_3D(model, samples: pandas.DataFrame, user_input=None, label=None, color_map=None, topn=5, sample=10):

    trees, websites, tld = samples['trees'], samples['url'], samples['tld']

    trees_tensor = Tensor(trees)
    tree_vectors = model.tree_model(trees_tensor).detach().numpy()

    three_dim = PCA(random_state=0).fit_transform(tree_vectors)[:, :2]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    for i, website in enumerate(websites):
        trace = go.Scatter3d(
            x=three_dim[i:i+1, 0],
            y=three_dim[i:i+1, 1],
            z=three_dim[i:i+1, 2],
            text=tld[i],
            name=website[-20:],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        data.append(trace)

    # trace_input = go.Scatter3d(
    #     x=three_dim[count:, 0],
    #     y=three_dim[count:, 1],
    #     z=three_dim[count:, 2],
    #     text=words[count:],
    #     name='input words',
    #     textposition="top center",
    #     textfont_size=20,
    #     mode='markers+text',
    #     marker={
    #         'size': 10,
    #         'opacity': 1,
    #         'color': 'black'
    #     }
    # )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    # data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()
    plotly.savefig('tree_embeddings.png')
    print('hello')


def display_pca_scatterplot_3D_vocab(model, words: List[str], vocab: Vocabulary):

    indexes = [vocab[word] for word in words]
    words_tensor = Tensor(indexes).long()
    word_vectors = model.tree_model.node_model.embedding(words_tensor).detach().numpy()

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :2]

    data = []
    for i, word in enumerate(words):
        trace = go.Scatter3d(
            x=three_dim[i:i+1, 0],
            y=three_dim[i:i+1, 1],
            z=three_dim[i:i+1, 2],
            text=word[-20:],
            name=word[-20:],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )
        data.append(trace)


    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )
    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()
    plotly.savefig('vocab_embeddings.png')
    print('hello')


if __name__ == '__main__':
    skip = 1
    stop = 1
    vocab = 0
    model_path = 'res_and_ckpts/base_small/checkpoints/pretrain/lstm.ckpt'
    df_path = 'data/updated_feather/dataframe'
    if vocab:
        model_path = 'res_and_ckpts/base_small/checkpoints/pretrain/lstm.ckpt'
        df_path = 'data/updated_feather/dataframe_vocab'

        skip = 0
        if not skip:
            print('loading model...')
            model = torch.load(model_path)

            print('loading vocab...')
            vocabulary = pickle_load('data/updated_feather/vocabs/total')
            print(len(vocabulary))

            word_list = []
            for key, value in vocabulary.items():
                if len(word_list) < 100 and key not in ['<mask>', '<oov>', '<ignore>']:
                    word_list.append(key)

            display_pca_scatterplot_3D_vocab(model=model, words=word_list, vocab=vocabulary)







    else:
        model_path = 'res_and_ckpts/transformer_attempts/checkpoints/pretrain/transformer.ckpt'
        df_path = 'data/updated_feather/dataframe'
        if not skip:
            print('loading model...')
            model = torch.load(model_path)

            print('loading feather... ')
            samples = pandas.read_feather('data/updated_feather/websites.feather').sample(frac=1)[:100]

            print('building strings...')
            strings = [file.decode('UTF-8', errors='ignore') for file in samples['html']]

            print('building trees...')
            trees, websites, tld = [], [], []
            for i, row in enumerate(list(samples.iterrows())):
                try:
                    row = row[1]
                    file, t, u = row['html'], row['tld'], row['url']
                    string = file.decode('UTF-8', errors='ignore')
                    trees.append(string_to_tree(string))
                    tld.append(t)
                    websites.append(u)
                except:
                    pass

            print('loading vocab...')
            vocabs = [pickle_load('data/updated_feather/vocabs/total')]
            print(len(vocabs[0]))

            print('reducing trees...')
            small_trees = [length_depth_reduction(tree, 1000, 5) for tree in trees]

            print('Tokenizing trees...')
            tree_tokenizer = TreeTokenizer(vocabs, total=True)
            tree_tokens = [tree_tokenizer(tree) for tree in trees]

            print('padding tokens...')
            tree_max = max(len(tree_token) for tree_token in tree_tokens)
            node_max = max(max(len(node_token) for node_token in tree_token) for tree_token in tree_tokens)
            for tree_token in tree_tokens:
                pad_tree(tree_token, tree_max, node_max)

            print('building df...')
            df = pandas.DataFrame()
            df['trees'] = tree_tokens
            df['tld'] = tld
            df['url'] = websites

            pickle_dump(df_path, df)

            if stop:
                raise Exception
        else:
            print('loading model...')
            model = torch.load(model_path)

            print('building df...')
            df = pickle_load(df_path)

        display_pca_scatterplot_3D(model=model, samples=df)


# Implemented following this site:
# https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5#:~:text=To%20visualize%20the%20word%20embedding%2C%20we%20are%20going%20to%20use,embedding%20GloVe%20will%20be%20implemented.
