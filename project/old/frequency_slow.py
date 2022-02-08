from typing import Dict
from code.parsing import *
import pickle
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
    def __init__(self, tf, af, df, kf, vf):
        super().__init__()
        self.tf = tf
        self.af = af
        self.df = df
        self.kf = kf
        self.vf = vf

    def handle_starttag(self, tag, attrs):
        self.tf.write(tag + " ")
        for attr in attrs:
            key, value = attr
            string = '(' + str(key) + ',' + str(value) + ')'
            self.af.write(string + " ")
            self.kf.write(str(key) + " ")
            self.vf.write(str(value) + " ")

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        data = data.strip()
        if data != "":
            self.df.write(data + " ")


def build_files(start_directory, end_directory):
    print('testing')
    with open(end_directory + "/tag_file.txt", 'w', errors='ignore') as tag_f, \
         open(end_directory + "/attr_file.txt", 'w', errors='ignore') as attr_f, \
         open(end_directory + "/key_file.txt", 'w', errors='ignore') as key_f, \
         open(end_directory + "/value_file.txt", 'w', errors='ignore') as value_f,  \
         open(end_directory + "/data_file.txt", 'w', errors='ignore') as data_f:

        strings = dir_to_str(start_directory)
        frequency_parser = FreqParser(tag_f, attr_f, data_f, key_f, value_f)
        [parse_string(x, frequency_parser) for x in strings]


def word_count(file_in, pickle_file):
    with open(file_in, 'r') as file:
        dictionary = {}
        words = file.read().split()
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        sorted_dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

        with open(pickle_file, 'wb') as handle:
            pickle.dump(sorted_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sorted_dictionary


def build_vocabs(directory='./common_sites', tag_floor=10, key_floor=10, value_floor=10, attr_floor=10):
    build_files(directory, './analysis')
    word_count('./analysis/attr_file.txt', './analysis/attr_pickled')
    word_count('../../setup/vocab/vocab_text_files/key_file.txt', './analysis/keys')
    word_count('../../setup/vocab/vocab_text_files/value_file.txt', './analysis/values')
    word_count('../../setup/vocab/vocab_text_files/tag_file.txt', './analysis/tags')
    tag_vocab, attr_vocab, key_vocab, value_vocab = Vocabulary(tag_floor, OTHER), Vocabulary(attr_floor, OTHER), Vocabulary(key_floor, OTHER), Vocabulary(value_floor, OTHER)
    with open('./analysis/attr_pickled', 'rb') as attr_pickle, \
         open('../../setup/vocab/frequency_dicts/tags', 'rb') as tag_pickle, \
         open('../../setup/vocab/frequency_dicts/keys', 'rb') as key_pickle, \
         open('../../setup/vocab/frequency_dicts/values', 'rb') as value_pickle:
        attrs, tags, keys, values = pickle.load(attr_pickle), pickle.load(tag_pickle), pickle.load(key_pickle), pickle.load(value_pickle)
        tag_vocab.feed(tags), attr_vocab.feed(attrs), key_vocab.feed(keys), value_vocab.feed(values)
        with open('vocabularies/vocabularies/tag_pickle', 'wb') as tag_vocab_pickle, \
             open('vocabularies/vocabularies/attr_pickle', 'wb') as attr_vocab_pickle, \
             open('vocabularies/vocabularies/key_pickle', 'wb') as key_vocab_pickle, \
             open('vocabularies/vocabularies/value_pickle', 'wb') as value_vocab_pickle:
            pickle.dump(tag_vocab, tag_vocab_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(attr_vocab, attr_vocab_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(key_vocab, key_vocab_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(value_vocab, value_vocab_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_trees(directory)


'''
        for tag in data:
            if data[tag] > sum(data)/100:
                vocabularies.append(tag)
        with open('./vocabulary/attrs_vocab', 'wb') as new_handle:
            pickle.dump(vocabularies, new_handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./analysis/keys','rb') as
'''


def pickle_trees(directory):
    strings = dir_to_str(directory)
    trees = [parse_string(html_string).tree for html_string in strings]
    with open('../../setup/trees/tree_directory', 'wb') as handle:
        pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)


# pickle_trees('./common_sites')
# rebuild()
# quick_analysis()
pickle_trees('./common_sites')

# build_files('./common_sites','./analysis')
# word_count('./analysis/tag_file.txt', './analysis/tags')
# word_count('./analysis/attr_file.txt', './analysis/attr_pickled')
# word_count('./analysis/key_file.txt', './analysis/keys')
# word_count('./analysis/value_file.txt', './analysis/values')
