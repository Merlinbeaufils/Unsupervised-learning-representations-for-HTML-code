import codecs
import os
import pickle
from html.parser import HTMLParser
from typing import List, Tuple

void_tags = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
             "source", "track", "wbr"]


class MyHTMLParser(HTMLParser):
    """
    Parser to build trees from html file
    """
    def __init__(self):
        super().__init__()
        # addressing omittable start tags
        self.tree: HtmlNode = HtmlNode("html", attrs=[], father=None, depth=0)
        self.head: HtmlNode = HtmlNode("head", attrs=[], father=self.tree, depth=1, child_index=0)
        self.body: HtmlNode = HtmlNode("body", attrs=[], father=self.tree, depth=1, child_index=1)
        self.tree.children = [self.body, self.head]
        self.node_stack = [self.tree, self.head]

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in ["html", "head", "body"]:
            father = self.node_stack[-1]
            node = HtmlNode(tag=tag.lower(), attrs=attrs, father=father, children=[], child_index=len(father.children))
            node.depth = len(self.node_stack)
            father.children.append(node)
            self.node_stack.append(node)
        elif tag.lower() == "body":
            self.node_stack.append(self.body)
            self.body.attrs = attrs

        elif tag.lower() == 'html':
            self.tree.attrs = attrs

        elif tag.lower() == 'head':
            self.head.attrs = attrs

        if tag in void_tags:
            self.node_stack.pop()

    def handle_endtag(self, tag):
        if tag == self.node_stack[-1].tag:
            self.node_stack.pop()
        else:
            print("fuck")
        return
        # print("End tag  :", tag)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in void_tags:
            self.handle_endtag(tag)
        return

    def handle_data(self, data):
        data = data.strip()
        if data and self.node_stack:
            # print("Data: ", data, "Node_stack: ",[node.show() for node in self.node_stack])
            self.node_stack[-1].data = data
        # node = self.node_stack[-1]
        # node.data = data
        # print(self.tree.tag == None)
        # print("Data     :", data)

    def handle_comment(self, data):
        pass
        # print("Comment  :", data)

    def handle_entityref(self, name):
        pass
        # c = chr(name2codepoint[name])
        # print("Named ent:", c)

    def handle_charref(self, name):
        pass
        # if name.startswith('x'):
        #    c = chr(int(name[1:], 16))
        # else:
        #    c = chr(int(name))
        # print("Num ent  :", c)

    def handle_decl(self, data):
        # print("Decl     :", data)
        pass


class HtmlNode:
    def __init__(self, tag="", attrs: List[Tuple[str, str]] = None,
                 father=None, children: List = None, depth=0, mask_val=0, child_index=0):
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
        if children is None:
            children = []
        if attrs is None:
            attrs = []
        self.father: HtmlNode = father
        self.children: List[HtmlNode] = children
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
            child.mask_affected()


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


def strings_to_trees(strings: List[str]) -> List[HtmlNode]:
    return [string_to_tree(string) for string in strings]


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


def pickle_dump(directory: str, item: any) -> None:
    """ Pickles a file into a given directory """
    with open(directory, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(directory: str) -> any:
    """ Loads a pickled file from a given directory """
    with open(directory, 'rb') as handle:
        try:
            return pickle.load(handle)
        except:
            print('Could not load handle: ', handle)
