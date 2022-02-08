import pickle
from argparse import Namespace
from html.parser import HTMLParser
from typing import Dict
import os

from project.current.parsing import dir_to_str, strings_to_trees, pickle_dump

OTHER: int = 0  # other value is currently 0

class Vocabulary(dict):  # Abstract dict class creating a tokenizing map from a frequency_dicts dictionary
    def __init__(self, floor: int = 100, other: int = OTHER):
        super().__init__()
        self.__setitem__('other', other)
        self.other = other
        self.vocab = ['other']
        self.floor = floor

    def __getitem__(self, item: str) -> int:
        if item in self.keys():
            return super().__getitem__(item)
        else:
            return self.other

    def feed(self, freq: Dict[str, int]):  # build map from frequency_dicts dict
        for key in freq:
            if freq[key] > self.floor and key not in self.keys():
                self.__setitem__(key, len(self.keys()))
                self.vocab.append(key)

    def reverse(self, val):
        return self.vocab[val]


class FreqParser(HTMLParser):
    def __init__(self, tf, df, kf, vf, total_f):
        super().__init__()
        self.tf = tf
        self.df = df
        self.kf = kf
        self.vf = vf
        self.total_f = total_f

    def handle_starttag(self, tag, attrs):
        self.tf.write(tag + " ")
        self.total_f.write(tag + " ")
        for attr in attrs:
            key, value = attr
            self.kf.write(str(key) + " ")
            self.vf.write(str(value) + " ")
            self.total_f.write(str(key) + " ")
            self.total_f.write(str(value) + " ")

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        data = data.strip()
        if data != "":
            self.df.write(data + " ")


def build_files(start_directory, end_directory) -> None:
    os.makedirs(end_directory, exist_ok=True)
    with open(end_directory + "/tags.txt", 'w', errors='ignore') as tag_f, \
         open(end_directory + "/keys.txt", 'w', errors='ignore') as key_f, \
         open(end_directory + "/values.txt", 'w', errors='ignore') as value_f,  \
         open(end_directory + "/data.txt", 'w', errors='ignore') as data_f, \
         open(end_directory + "/total.txt", 'w', errors='ignore') as total_f:

        strings = dir_to_str(start_directory)
        frequency_parser = FreqParser(tag_f, data_f, key_f, value_f, total_f)
        [frequency_parser.feed(string) for string in strings]


def build_trees(directory, args: Namespace) -> None:
    strings = dir_to_str(directory)
    trees = strings_to_trees(strings)
    args.trees = trees
    os.makedirs(directory + 'trees', exist_ok=True)
    if args.pickle_trees:
        pickle_dump(directory + 'trees/trees', trees)


def word_count(file_in: str, pickle_file: str) -> Dict:
    with open(file_in, 'r') as file:
        dictionary = {}
        words = file.read().split()
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        # noinspection PyTypeChecker
        sorted_dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
        pickle_dump(pickle_file, sorted_dictionary)
        return sorted_dictionary


def build_vocabs(directory, tag_floor=10, key_floor=10, value_floor=10, total_floor=10, args: Namespace = None):
    build_files(directory, directory + 'text_files')
    os.makedirs(directory + 'frequency_dict', exist_ok=True)
    os.makedirs(directory + 'vocabs', exist_ok=True)
    keys = word_count(directory + 'text_files/keys.txt', directory + 'frequency_dict/key_freq')
    values = word_count(directory + 'text_files/values.txt', directory + 'frequency_dict/values_freq')
    tags = word_count(directory + 'text_files/tags.txt', directory + 'frequency_dict/tags_freq')
    total = word_count(directory + 'text_files/total.txt', directory + 'frequency_dict/total_freq')
    tag_vocab, key_vocab = Vocabulary(tag_floor, OTHER), Vocabulary(key_floor, OTHER)
    total_vocab, value_vocab = Vocabulary(total_floor, OTHER), Vocabulary(value_floor, OTHER)
    tag_vocab.feed(tags), key_vocab.feed(keys), value_vocab.feed(values), total_vocab.feed(total)
    args.tags, args.keys, args.values, args.total = tag_vocab, key_vocab, value_vocab, total_vocab
    args.tags['mask'] = len(tags)
    args.total['mask'] = len(total)
    pickle_dump(directory + 'vocabs/tags', args.tags), pickle_dump(directory + 'vocabs/keys', args.keys)
    pickle_dump(directory + 'vocabs/values', args.values), pickle_dump(directory + 'vocabs/total', args.total)


def pickle_trees(directory):
    strings = dir_to_str(directory)
    trees = strings_to_trees(strings)
    pickle_dump(directory + '/trees/trees', trees)


# pickle_trees('./common_sites')
# rebuild()
# quick_analysis()

# build_files('./common_sites','./analysis')
# word_count('./analysis/tag_file.txt', './analysis/tags')
# word_count('./analysis/key_file.txt', './analysis/keys')
# word_count('./analysis/value_file.txt', './analysis/values')
