import os
from html.parser import HTMLParser
from typing import Dict
from typing import Tuple, List

from torch import LongTensor
from project.parsing import dir_to_str, strings_to_trees, pickle_dump, HtmlNode, pandas_to_strings, \
    strings_to_trees_and_files
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
FILE_SPLIT_TOKEN = '<<<START_OF_HTML_FILE>>>'

OTHER: int = 0  # other value is currently 0


class Vocabulary(dict):  # Abstract dict class creating a tokenizing map from a frequency_dicts dictionary
    """
    Slightly more complex dict class to be used as a vocabulary

    Returns out of vocab value when key is not found.

    Automatically implements oov, ignore and mask tokens
    """
    def __init__(self, freq: dict, floor: int = 10):
        super().__init__()
        self.__setitem__('<ignore>', 0)
        self.__setitem__('<oov>', 1)
        self.__setitem__('<mask>', 2)

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
        return list(self.keys())[val]

    def reverse_vocab(self) -> Dict:
        return {value: key for key, value in self.items()}


class FreqParser(HTMLParser):
    """
    Builds text of tags, data, keys, values and everything-combined while parsing html file.
    """
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


def build_files(start_directory, end_directory, key_only=False, pandas=False) -> None:
    """
    Builds files for building vocabs and frequency analysis

    :param start_directory: takes html files from this directory
    :param end_directory: places text files at this directory
    :param key_only: disregards "values" totally (reduces complexity of data)
    :param pandas: if True: pandas file, else directory of strings.
    :return: None
    """
    os.makedirs(end_directory, mode=0o777, exist_ok=True)
    with open(end_directory + "/tags.txt", 'w', errors='ignore') as tag_f, \
         open(end_directory + "/keys.txt", 'w', errors='ignore') as key_f, \
         open(end_directory + "/values.txt", 'w', errors='ignore') as value_f,  \
         open(end_directory + "/data.txt", 'w', errors='ignore') as data_f, \
         open(end_directory + "/total.txt", 'w', errors='ignore') as total_f:

        strings = dir_to_str(start_directory) if not pandas else pandas_to_strings(start_directory)
        frequency_parser = FreqParser(tag_f, data_f, key_f, value_f, total_f, key_only)
        [frequency_parser.feed(string) for string in strings]


def build_trees(directory, pickle_trees: bool = False, pandas: bool = False) -> List[HtmlNode]:
    """
    Builds trees from the given directory
    :param directory: html file directory
    :param pickle_trees: Pickle trees into memory at directory/trees/trees
    :param pandas: if True: pandas file of html strings, else: directory of html strings
    :return: List of trees as HtmlNodes
    """
    strings = dir_to_str(directory) if not pandas else pandas_to_strings(directory)
    trees = strings_to_trees(strings)
    os.makedirs(directory + 'trees', mode=0o777, exist_ok=True)
    if pickle_trees:
        pickle_dump(directory + 'trees/trees', trees)
    return trees


def build_trees_and_files(directory, pandas: bool = False, max_trees=1000):
    print(pandas)
    strings = dir_to_str(directory) if not pandas else pandas_to_strings(directory, max_trees * 3)
    directory = directory if not pandas else directory + '/'
    trees = strings_to_trees_and_files(strings, directory, max_trees)
    os.makedirs(directory + 'trees', mode=0o777, exist_ok=True)
    pickle_dump(directory + 'trees/trees', trees)
    return trees


def word_count(file_in: str, pickle_file: str) -> Dict[str, int]:
    """
    Creates frequency dictionaries from the built text file
    :param file_in: Text file to process
    :param pickle_file: pickle at this directory
    :return: Ordered dictionary with frequencies of words in the text file
    """
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
    """
    Make sure to build text files before
    Builds vocabularies from directory of text files.
    :param directory:
    :param tag_floor: minimum for tags
    :param key_floor: minimum for keys
    :param value_floor: minimum for values
    :param total_floor: minimum for total_vocab
    :return: Tag vocabuly, key vocabulary, value vocabulary, total vocabulry
    """
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


def term_frequency(data_file):
    file = codecs.open(data_file, "r", "utf-8", errors='ignore')
    strings = file.read().strip(FILE_SPLIT_TOKEN).split(FILE_SPLIT_TOKEN)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(strings)
    return matrix, vectorizer


def pickle_trees(directory):
    strings = dir_to_str(directory)
    trees = strings_to_trees(strings)
    pickle_dump(directory + '/trees/trees', trees)


def test():
    trees = build_trees_and_files(directory='data/feather', pandas=True, max_trees=10)
    matrix, vectorizer = term_frequency('data/feather/text_files/data.txt')
    print('hi')


#test()
