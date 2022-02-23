import os
from html.parser import HTMLParser
from typing import Dict
from typing import Tuple, List
from torch import LongTensor

from project.parsing import dir_to_str, strings_to_trees, pickle_dump, HtmlNode

OTHER: int = 0  # other value is currently 0


class Vocabulary(dict):  # Abstract dict class creating a tokenizing map from a frequency_dicts dictionary
    def __init__(self, freq: dict, floor: int = 10):
        super().__init__()
        self.__setitem__('<oov>', 0)
        self.__setitem__('<ignore>', 1)
        self.__setitem__('<mask>', 2)
        #self.__setitem__('<SON>', 3)
        # self.__setitem__('<EON>', 4)
        self.floor = floor
        self.frequency = freq
        self.feed(freq)

    def __getitem__(self, item: str) -> int:
        if item in self.keys():
            return super().__getitem__(item)
        else:
            return super().__getitem__('<oov>')

    def feed(self, freq: Dict[str, int]) -> None:  # build map from frequency_dicts dict
        for key in freq:
            if freq[key] > self.floor and key not in self.keys():
                self.__setitem__(key, len(self.keys()))

    def reverse(self, val: LongTensor) -> str:
        return list(self.frequency.keys())[val]

    def reverse_vocab(self) -> Dict:
        return {value: key for key, value in self.items()}


class FreqParser(HTMLParser):
    def __init__(self, tf, df, kf, vf, total_f, key_only=False):
        super().__init__()
        self.tf = tf
        self.df = df
        self.kf = kf
        self.vf = vf
        self.total_f = total_f
        self.key_only = key_only

    def handle_starttag(self, tag, attrs):
        self.tf.write(tag + " ")
        self.total_f.write(tag + " ")
        for attr in attrs:
            key, value = attr
            self.kf.write(str(key) + " ")
            self.vf.write(str(value) + " ")
            self.total_f.write(str(key) + " ")
            if not self.key_only:
                self.total_f.write(str(value) + " ")

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        data = data.strip()
        if data != "":
            self.df.write(data + " ")


def build_files(start_directory, end_directory, key_only=False) -> None:
    os.makedirs(end_directory, mode=0o777, exist_ok=True)
    with open(end_directory + "/tags.txt", 'w', errors='ignore') as tag_f, \
         open(end_directory + "/keys.txt", 'w', errors='ignore') as key_f, \
         open(end_directory + "/values.txt", 'w', errors='ignore') as value_f,  \
         open(end_directory + "/data.txt", 'w', errors='ignore') as data_f, \
         open(end_directory + "/total.txt", 'w', errors='ignore') as total_f:

        strings = dir_to_str(start_directory)
        frequency_parser = FreqParser(tag_f, data_f, key_f, value_f, total_f, key_only)
        [frequency_parser.feed(string) for string in strings]


def build_trees(directory, Pickle_trees: bool=False) -> List[HtmlNode]:
    strings = dir_to_str(directory)
    trees = strings_to_trees(strings)
    os.makedirs(directory + 'trees', mode=0o777, exist_ok=True)
    if Pickle_trees:
        pickle_dump(directory + 'trees/trees', trees)
    return trees


def word_count(file_in: str, pickle_file: str) -> Dict[str, int]:
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


def build_vocabularies(directory, tag_floor=2, key_floor=2, value_floor=2, total_floor=10)\
        -> Tuple[Vocabulary, Vocabulary, Vocabulary, Vocabulary]:
    os.makedirs(directory + 'frequency_dict', mode=0o777, exist_ok=True)
    os.makedirs(directory + 'vocabs', mode=0o777, exist_ok=True)
    keys = word_count(directory + 'text_files/keys.txt', directory + 'frequency_dict/key_freq')
    values = word_count(directory + 'text_files/values.txt', directory + 'frequency_dict/values_freq')
    tags = word_count(directory + 'text_files/tags.txt', directory + 'frequency_dict/tags_freq')
    total = word_count(directory + 'text_files/total.txt', directory + 'frequency_dict/total_freq')
    tag_vocab, key_vocab = Vocabulary(tags, tag_floor), Vocabulary(keys, key_floor)
    total_vocab, value_vocab = Vocabulary(total, total_floor), Vocabulary(values, value_floor)
    pickle_dump(directory + 'vocabs/tags', tag_vocab), pickle_dump(directory + 'vocabs/keys', key_vocab)
    pickle_dump(directory + 'vocabs/values', value_vocab), pickle_dump(directory + 'vocabs/total', total_vocab)
    return tag_vocab, key_vocab, value_vocab, total_vocab


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
