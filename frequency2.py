from typing import Dict
from parsing import *
import pickle
OTHER: int = 0  # other value is currently 0


class Vocabulary(dict):  # Abstract dict class creating a tokenizing map from a frequency dictionary
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

    def feed(self, freq: Dict[str, int]):  # build map from frequency dict
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
    print('testing')
    with open(end_directory + "/tag_file.txt", 'w', errors='ignore') as tag_f, \
         open(end_directory + "/key_file.txt", 'w', errors='ignore') as key_f, \
         open(end_directory + "/value_file.txt", 'w', errors='ignore') as value_f,  \
         open(end_directory + "/data_file.txt", 'w', errors='ignore') as data_f, \
         open(end_directory + "/total_file.txt", 'w', errors='ignore') as total_f:

        strings = dir_to_str(start_directory)
        frequency_parser = FreqParser(tag_f, data_f, key_f, value_f, total_f)
        [parse_string(x, frequency_parser) for x in strings]


def build_trees(directory, args: Namespace) -> None:
    strings = dir_to_str(directory)
    trees = [parse_string(string).tree for string in strings]
    args.trees = trees
    if args.pickle_trees:
        with open('./tree_directory', 'wb') as handle:
            pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)


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

        with open(pickle_file, 'wb') as handle:
            pickle.dump(sorted_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sorted_dictionary


def build_vocabs(directory='./common_sites', tag_floor=10, key_floor=10, value_floor=10, total_floor=10, args: Namespace = None):
    build_files(directory, './analysis')
    keys = word_count('./analysis/key_file.txt', './analysis/key_freq')
    values = word_count('./analysis/value_file.txt', './analysis/value_freq')
    tags = word_count('./analysis/tag_file.txt', './analysis/tag_freq')
    total = word_count('./analysis/total_file.txt', './analysis/total_freq')
    tag_vocab, key_vocab = Vocabulary(tag_floor, OTHER), Vocabulary(key_floor, OTHER)
    total_vocab, value_vocab = Vocabulary(total_floor, OTHER), Vocabulary(value_floor, OTHER)
    tag_vocab.feed(tags), key_vocab.feed(keys), value_vocab.feed(values), total_vocab.feed(total)
    args.tags, args.keys, args.values, args.total = tag_vocab, key_vocab, value_vocab, total_vocab
    args.tags['mask'] = len(tags)
    args.total['mask'] = len(total)
    pickle_dump('./vocab/tags', args.tags), pickle_dump('./vocab/keys', args.keys)
    pickle_dump('./vocab/values', args.values), pickle_dump('./vocab/total', args.total)







def pickle_trees(directory):
    strings = dir_to_str(directory)
    trees = [parse_string(html_string).tree for html_string in strings]
    with open('./tree_directory', 'wb') as handle:
        pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)


# pickle_trees('./common_sites')
# rebuild()
# quick_analysis()

# build_files('./common_sites','./analysis')
# word_count('./analysis/tag_file.txt', './analysis/tag_pickled')
# word_count('./analysis/key_file.txt', './analysis/key_pickled')
# word_count('./analysis/value_file.txt', './analysis/value_pickled')
