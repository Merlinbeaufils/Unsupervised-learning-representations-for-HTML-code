import codecs
import math
import os
import pickle
from html.parser import HTMLParser
from typing import List, Tuple

import numpy as np
import pandas

sep_token = ")*&&SEPARATION)*&&"

dont_handle_data = ['script', 'style']
do_handle_data = ['p', 'span', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'blockquote', 'q',
                  'li', 'dt', 'dd', 'mark', 'ins', 'del', 'sup', 'sub', 'small', 'i', 'b']
current_amount_of_pickled_trees = 28924

void_tags = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
             "source", "track", "wbr"]


class MyHTMLParser(HTMLParser):
    """
    Parser to build trees from html file
    """
    def __init__(self, files=None, key_only: bool = False):
        super().__init__()
        # addressing omittable start tags
        self.tree: HtmlNode = HtmlNode("html", attrs=[], father=None, depth=0)
        self.head: HtmlNode = HtmlNode("head", attrs=[], father=self.tree, depth=1, child_index=0, root_node=self.tree)
        self.body: HtmlNode = HtmlNode("body", attrs=[], father=self.tree, depth=1, child_index=1, root_node=self.tree)
        self.tree.children = [self.head, self.body]
        self.node_stack = [self.tree, self.head]
        self.build_files = 0
        if files is not None:
            self.build_files = 1
            self.tag_file, self.key_file, self.value_file, self.data_file, \
                self.total_file, self.depth_file, self.node_file = files
        self.key_only = key_only

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in ["html", "head", "body"]:
            father = self.node_stack[-1]
            node = HtmlNode(tag=tag.lower(), attrs=attrs, father=father,
                            children=[], child_index=len(father.children), root_node=self.tree)
            node.depth = len(self.node_stack)
            father.children.append(node)
            self.node_stack.append(node)
        elif tag.lower() == "body":
            self.node_stack.append(self.body)
            self.body.attrs = attrs
            node = self.body

        elif tag.lower() == 'html':
            self.tree.attrs = attrs
            node = self.tree

        elif tag.lower() == 'head':
            self.head.attrs = attrs
            node = self.head

        if tag in void_tags:
            self.node_stack.pop()

        if self.build_files:
            self.tag_file.write(tag + sep_token)
            self.depth_file.write(str(node.depth) + sep_token)
            self.node_file.write(str(tag) + str(attrs) + sep_token)
            self.total_file.write(tag + sep_token)
            for attr in attrs:
                key, value = attr
                self.key_file.write(str(key) + sep_token)
                self.value_file.write(str(value) + sep_token)
                self.total_file.write(str(key) + sep_token)
                if not self.key_only:
                    self.total_file.write(str(value) + sep_token)

    def handle_endtag(self, tag):
        if tag not in void_tags:
            if tag == self.node_stack[-1].tag:
                self.node_stack.pop()
            else:
                self.node_stack.pop()
                self.handle_endtag(tag)

        if self.build_files and tag == 'html':
            self.data_file.write('<<<START_OF_HTML_FILE>>>')

    def handle_data(self, data):
        data = data.strip()
        curr_node = None if not self.node_stack else self.node_stack[-1]
        tag = False if not self.node_stack else curr_node.tag
        handle_data = False if not self.node_stack else tag in do_handle_data

        if data and handle_data:
            curr_node.data += data
            if self.build_files:
                self.data_file.write(tag + ':' + data + sep_token)


class HtmlNode:
    def __init__(self, tag="", attrs: List[Tuple[str, str]] = None,
                 father=None, children: List = None, depth=0, mask_val=0,
                 child_index=0, root_node=None):
        """
        Tree structure to represent files
        :param tag: element tag
        :param attrs: list of (key,value) attribute pairs
        :param father: father node
        :param children: list of children nodes
        :param depth: depth in the tree
        :param mask_val: Denotes whether this node was masked
        :param child_index: Index of child in parent node
        """

        self.children: List[HtmlNode] = [] if children is None else children
        self.attrs = [] if attrs is None else attrs
        self.root_node = self if root_node is None else root_node

        self.father: HtmlNode = father
        self.tag: str = tag
        self.attrs = attrs
        self.data: str = ""
        self.depth: int = depth
        self.sim_string = 1
        self.path: List[HtmlNode] = []
        self.mask_val: int = mask_val
        self.temp_tag, self.temp_attrs, self.temp_data, self.temp_children = '', [], '', []
        self.child_index = child_index
        # self.total_children = 0
        # self.next = [None for x in range(self.total_children)]
        # self.leaf = False

    def __getitem__(self, tag):
        return getattr(self, tag)

    def __setitem__(self, tag, node):
        setattr(self, tag, node)

    '''
    def __str__(self):
        indent = " " * 4 * self.depth
        children_str = "\n".join(map(str, self.children))
        if children_str:
            children_str = "\n" + children_str
        if self.sim_string:
            return indent + "Node: %s, data:%s %s" % (self.tag, self.data, children_str)
        else:
            return indent + "Node: %s, attrs: %s, data: %s,  depth: %s %s" % (
                self.tag, self.attrs, self.data, self.depth, children_str)
    '''
    def __str__(self):
        # root_depth = self.depth if root_depth == -1 else root_depth
        indent = " " * 4 * self.depth
        children_str = "\n".join(map(str, self.children))
        if children_str:
            children_str = "\n" + children_str
        if self.sim_string:
            attributes = " ".join(['%s="%s"' % (key, value) for key, value in self.attrs])
            return indent + "<%s %s> %s %s </%s>" % (self.tag, attributes, self.data.strip(), children_str, self.tag)

    def __len__(self):
        return len(self.path)

    def __bool__(self):
        return bool(self.children)

    def show(self) -> str:
        # Recursively build up the full string
        if self.children:
            return self.tag + ", " + self.data
            # return f'({self.tag,self.data} {" ".join(str(child) for child in self.children)})'
        # Base case - no children; Just return the tag.
        else:
            return self.tag

    def build_path(self):
        """
        Builds the post-order path through the tree
        :return: None
        """
        path: List[HtmlNode] = []
        if not self.children:
            self.path = [self]
            return self.path
        else:
            for child in self.children:
                path += child.build_path()
            path += [self]
            self.path = path
            return self.path

    def mask(self, tree_path_index):  # Dont use
        """ Don't use this """
        target_node = self.path[tree_path_index]
        self.masked_father = target_node.father
        self.masked_child_index = target_node.child_index

        target_node.mask_self()


    def mask_self(self):
        self.temp_tag = self.tag
        self.temp_attrs = self.attrs.copy()
        self.temp_data = self.data
        self.temp_children = self.children
        self.tag = '<mask>'
        self.attrs.clear()
        self.data = ''
        self.mask_val = 1
        self.children = []

    def unmask_self(self):
        self.tag = self.temp_tag
        self.data = self.temp_data
        self.attrs = self.temp_attrs
        self.temp_tag = ''
        self.temp_data = ''
        self.temp_attrs = []
        self.mask_val = 0

    def affected(self) -> List:
        """ return list of dependent nodes """
        full = [self]
        for child in self.children:
            full += child.affected()
        return full

    def mask_affected(self):
        """ mask all affected nodes including itself """
        self.mask_val = 1
        for child in self.children:
            child.mask_affected()

    def unmask_affected(self):
        """ unmask all affected nodes including itself """
        self.mask_val = 0
        for child in self.children:
            child.unmask_affected()




def string_to_tree(string: str) -> HtmlNode:
    """
    Build tree from a string
    :param string: html string
    :return: Node representing tree
    """
    parser = MyHTMLParser()
    parser.feed(string)
    parser.tree.build_path()
    [parser.tree.children.remove(x) for x in parser.tree.children if x.tag.lower() not in ['head', 'body']]
    return parser.tree


def strings_to_trees_and_files(strings: List[str], directory: str, max_trees=1000, framework='pretrain'):
    # panda_dir = directory + '/masked_websites.feather'
    os.makedirs(directory + '/text_files_' + framework, mode=0o777, exist_ok=True)
    directory = directory + '/text_files_' + framework
    with open(directory + "/tags.txt", 'w', errors='ignore') as tag_f, \
            open(directory + "/keys.txt", 'w', errors='ignore') as key_f, \
            open(directory + "/values.txt", 'w', errors='ignore') as value_f, \
            open(directory + "/data.txt", 'w', errors='ignore') as data_f, \
            open(directory + "/total.txt", 'w', errors='ignore') as total_f, \
            open(directory + "/depth.txt", 'w', errors='ignore') as depth_f, \
            open(directory + "/node.txt", 'w', errors='ignore') as node_f:
        trees = []
        i = 0
        while len(trees) < max_trees and i < len(strings):
            string = strings[i]
            parser = MyHTMLParser([tag_f, key_f, value_f, data_f, total_f, depth_f, node_f])
            try:
                parser.feed(string)
                tree = parser.tree
                tree.build_path()
                trees.append(tree)
                if not len(trees) % 200:
                    print("Gone through: ", i,' and length of trees is: ', len(trees))
            except Exception:
                print("THATS NO BUENO~~~~~~~~~~~~~~~~~~~~~")
            i += 1
    return trees

def strings_to_trees(strings: List[str], describe=True, index=0) -> List[HtmlNode]:
    if describe:
        trees = []
        non_working_files = []
        i = index
        while len(trees) < 1000 and i < len(strings):
            try:
                trees.append(string_to_tree(strings[i]))
            except Exception:
                print('File #: ', i)
                non_working_files.append(i)
            i += 1
        pickle_dump('data/feather/tree_batches/' + str(index), trees)
        strings_to_trees(strings, True, i)
    else:
        return [string_to_tree(string) for string in strings]


def pickle_panda_strings(directory: str):
    panda_file = pandas.read_feather(directory)
    html_files = panda_file['html']
    strings = [file.decode('UTF-8', errors='ignore') for file in html_files]
    pickle_dump('data/feather/strings', strings)


def pickle_panda_trees_from_string(directory='data/feather/strings'):
    strings = pickle_load(directory)
    index = 0
    non_working = []
    some_val = 2
    while index < some_val:
        temp_non_working, i, trees = pickle_batch_of_trees(strings, index)
        with open('data/feather/tree_batches/' + str(index), 'wb') as handle:
            pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)
        non_working += temp_non_working
        index = i
    pickle_dump('data/feather/non_working', non_working)


def pickle_batch_of_trees(strings, index):
    trees, non_working = [], []
    i = index
    while len(trees) < 1000 and i < len(strings):
        try:
            trees.append(string_to_tree(strings[i]))
        except Exception:
            # print('File #: ', i)
            non_working.append(i)
        i += 1
    print('done with batch and index = ', i)
    return non_working, i, trees


def dir_to_str(directory: str) -> [str]:
    """
    Builds list of strings from directory of html files
    :param directory: directory of html files
    :return: list of strings
    """
    strings = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f, "bruh")
            file = codecs.open(f, "r", "utf-8")
            strings.append(file.read())
    return strings


def pandas_to_strings(directory: str, max_strings=3000, framework='pretrain') -> [str]:
    panda_directory = directory + 'final_data_' + framework + '.csv.gz'
    panda_file = pandas.read_csv(panda_directory, compression='gzip')
    panda_file = panda_file.sample(frac=1)
    panda_file = panda_file.reset_index()[['html', 'tld', 'url']]
    panda_file = panda_file[:max_strings]
    return panda_file['html']
    # return [file.decode('UTF-8', errors='ignore') for file in html_files]


def pickle_dump(directory: str, item: any) -> None:
    """ Pickles a file into a given directory """
    with open(directory, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(directory: str) -> any:
    """ Loads a pickled file from a given directory """
    with open(directory, 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception:
            print('Could not load handle: ', handle)



def test():
    dir = 'data/random/text_files'
    with open(dir + "/tags.txt", 'w', errors='ignore') as tag_f, \
            open(dir + "/keys.txt", 'w', errors='ignore') as key_f, \
            open(dir + "/values.txt", 'w', errors='ignore') as value_f, \
            open(dir + "/data.txt", 'w', errors='ignore') as data_f, \
            open(dir + "/total.txt", 'w', errors='ignore') as total_f, \
            open(dir + "/depth.txt", 'w', errors='ignore') as depth_f, \
            open(dir + "/node.txt", 'w', errors = 'ignore') as node_f:

        strings = dir_to_str('data/random')
        for string in strings:
            parser = MyHTMLParser([tag_f, key_f, value_f, data_f, total_f, depth_f, node_f])
            parser.feed(string)

        print('hi')


#test()
