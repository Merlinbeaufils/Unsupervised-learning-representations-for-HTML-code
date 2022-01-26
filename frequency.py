from parsing import *
from html.parser import HTMLParser
# import codecs
# import os
import seaborn as sns
import parsing
import pickle


class FreqParser(HTMLParser):
    def __init__(self, tf, af, df):
        super().__init__()
        self.tf = tf
        self.af = af
        self.df = df

    def handle_starttag(self, tag, attrs):
        self.tf.write(tag + " ")
        for attr in attrs:
            string = str(attr)
            index = string.find(',')
            string = string[:index + 1] + string[index + 2:]
            self.af.write(string + " ")

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        data = data.strip()
        if data != "":
            self.df.write(data + " ")


def build_files(start_directory, end_directory):
    print('testing')
    with open(end_directory + "/tag_file.txt", 'w', errors='ignore') as tf, \
         open(end_directory + "/attrs_file.txt", 'w', errors='ignore') as af, \
         open(end_directory + "/data_file.txt", 'w', errors='ignore') as df:

        strings = parsing.dir_to_str(start_directory)
        parser = FreqParser(tf, af, df)
        [parsing.parse_string(x, parser) for x in strings]


def word_count(file_in, pickle_file):
    with open(file_in, 'r') as file:
        dictionary = {}
        words = file.read().split()
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        sorted_dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]))
        with open(pickle_file, 'wb') as handle:
            pickle.dump(sorted_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sorted_dictionary


def trim_dict(dictionary, floor):
    print(len(dictionary))
    delete = []
    for x in dictionary:
        if dictionary[x] < floor:
            delete.append(x)
    for x in delete:
        del dictionary[x]
    print(len(dictionary))


def analyze_results(file_in, file_out, pikl=0, floor=0, scale='linear'):
    if pikl == 1:
        with open(file_in[:-4] + "_pickled", 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = word_count(file_in, file_in[:-4] + '_pickled')

    trim_dict(data, floor)
    plot = sns.barplot(x=list(data.keys()), y=list(data.values()))
    plot.set_yscale(scale)
    fig = plot.get_figure()
    fig.savefig(file_out)


# analyze_results(path + 'tag_file.txt', path + 'plots/tag_count.png', pikl=1, floor=2, scale='log')
# analyze_results(path + 'attrs_file.txt', path + 'plots/attrs_count.png')
# analyze_results(path + 'data_file.txt', path + 'plots/data_count.png')
# dict = word_count(path + 'attrs_file.txt', path + 'attrs_file_pickled')


def rebuild():
    path = './analysis/'
    build_files('./common_sites', './analysis')
    word_count(path + 'tag_file.txt', path + 'tag_file_pickled')
    word_count(path + 'attrs_file.txt', path + 'attrs_file_pickled')


def quick_analysis():
    path = './analysis/'
    analyze_results(path + 'tag_file.txt', path + 'plots/' + 'tag_count.png', pikl=1, floor=50, scale='log')
    analyze_results(path + 'attrs_file.txt', path + 'plots/' + 'attrs_count.png', pikl=1, floor=100, scale='linear')


def build_tag_vocabulary(directory):
    build_files(directory, './analysis/tag_file.txt')
    word_count('./analysis/tag_file.txt', './analysis/tag_pickled')
    vocab = []
    with open('./analysis/tag_pickled', 'rb') as handle:
        data = pickle.load(handle)
        for tag in data:
            if data[tag] > sum(data)/100:
                vocab.append(tag)
        with open('./vocabulary/tag_vocab', 'wb') as new_handle:
            pickle.dump(vocab, new_handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_attrs_vocabulary(directory):
    build_files(directory, './analysis/attrs_file.txt')
    word_count('./analysis/attrs_file.txt', './analysis/attrs_pickled')
    vocab = []
    with open('./analysis/attrs_pickled', 'rb') as handle:
        data = pickle.load(handle)
        for tag in data:
            if data[tag] > sum(data)/100:
                vocab.append(tag)
        with open('./vocabulary/attrs_vocab', 'wb') as new_handle:
            pickle.dump(vocab, new_handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_trees(directory):
    strings = dir_to_str(directory)
    trees = [parse_string(html_string, MyHTMLParser()).tree for html_string in strings]
    with open('./tree_directory', 'wb') as handle:
        pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)


pickle_trees('./common_sites')
# rebuild()
# quick_analysis()


# build_files("./common_sites", "./analysis")

