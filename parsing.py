from html.parser import HTMLParser
import os
# from html.entities import name2codepoint
import codecs
from typing import List, Tuple

void_tags = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
             "source", "track", "wbr"]


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # addressing omittable start tags
        self.head = HtmlNode("head", [], None, [], depth=1)
        self.body = HtmlNode("body", [], None, [], depth=1)
        self.tree = HtmlNode("html", [], None, [self.head, self.body])
        self.head.father = self.body.father = self
        self.node_stack = [self.tree, self.head]

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in ["html", "head", "body"]:
            father = self.node_stack[-1]
            node = HtmlNode(tag=tag.lower(), attrs=attrs, father=father, children=[])
            node.depth = len(self.node_stack)
            father.children.append(node)
            self.node_stack.append(node)
        elif tag.lower() == "body":
            self.node_stack.append(self.body)

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
    def __init__(self, tag="", attrs: Tuple[str, str] = None, father=None, children: List = None, depth=0, mask_val=0):
        if children is None:
            children = []
        if attrs is None:
            attrs = []
        self.father = father
        self.children = children
        self.tag = tag
        self.attrs = attrs
        self.data = ""
        self.depth = depth
        self.sim_string = 1
        self.path = []
        self.mask_val = mask_val
        self.temp_tag, self.temp_attrs, self.temp_data = '', [], ''
        # self.total_children = 0
        # self.next = [None for x in range(self.total_children)]
        # self.leaf = False

    def __getitem__(self, tag):
        return getattr(self, tag)

    def __setitem__(self, tag, node):
        setattr(self, tag, node)

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
        path = []
        if not self.children:
            self.path = [self]
            return self.path
        else:
            for child in self.children:
                path += child.build_path()
            path += [self]
            self.path = path
            return self.path

    def mask(self, tree_path_index):
        self.path[tree_path_index].mask_self()

    def mask_self(self):
        self.temp_tag = self.tag
        self.temp_attrs = self.attrs.copy()
        self.temp_data = self.data
        self.tag = "mask"
        self.attrs.clear()
        self.data = ''
        self.mask_val = 1

    def unmask_self(self):
        self.tag = self.temp_tag
        self.data = self.temp_data
        self.attrs = self.temp_attrs
        self.temp_tag = ''
        self.temp_data = ''
        self.temp_attrs = []
        self.mask_val = 0


def parse_string(string: str, parser=None):
    t = 0
    if parser is None:
        parser = MyHTMLParser()
        t = 1
    x = parser
    x.feed(string)
    if t == 1:
        x.tree.build_path()
    return x


def dir_to_str(directory: str) -> [str]:
    strings = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f, "bruh")
            file = codecs.open(f, "r", "utf-8")
            strings.append(file.read())
    return strings
