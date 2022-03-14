from typing import List


from sklearn.feature_extraction.text import TfidfVectorizer
from project.parsing import pickle_load


def build_transform_from_list_of_strings(strings, dir=True):
    """
    Builds tf-idf transformation matrix from list of strings
    :param strings: directory of list of strings or list of strings
    :param dir: determines if strings is a directory
    :return:
    """
    if dir:
        strings: List[str] = pickle_load(strings)
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(strings)
