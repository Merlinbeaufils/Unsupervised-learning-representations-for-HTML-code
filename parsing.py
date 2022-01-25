from html.parser import HTMLParser
import os
# from html.entities import name2codepoint
import codecs


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tree = HtmlNode("root", [], None, [])
        self.node_stack = [self.tree]

    def handle_starttag(self, tag, attrs):
        if tag == "html":
            self.tree = HtmlNode(tag=tag, attrs=attrs, father=None, children=[])
            self.tree.depth = 0
            self.node_stack = [self.tree]
        else:
            father = self.node_stack[-1]
            node = HtmlNode(tag=tag, attrs=attrs, father=father, children=[])
            node.depth = len(self.node_stack)
            father.children.append(node)
            self.node_stack.append(node)
        #self.tree.total_children += 1
        # print("Start tag:", tag)
        # for attr in attrs:
            # print("     attr:", attr)

    def handle_endtag(self, tag):
        node = self.node_stack.pop()
        node.active = False
        # print("End tag  :", tag)

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
    def __init__(self, tag="", attrs=None, father=None, children=None):
        if children is None:
            children = []
        if attrs is None:
            attrs = []
        self.father = father
        self.children = children
        self.tag = tag
        self.attrs = attrs
        self.active = True
        self.data = ""
        self.depth = 0
        self.sim_string = 1
        #self.total_children = 0
        #self.next = [None for x in range(self.total_children)]
        #self.leaf = False

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

    def show(self) -> str:
        # Recursively build up the full string
        if self.children:
            return self.tag + ", " + self.data
            # return f'({self.tag,self.data} {" ".join(str(child) for child in self.children)})'
        # Base case - no children; Just return the tag.
        else:
            return self.tag


def parse_string(string: str, parser: HTMLParser) -> HtmlNode:
    x = parser
    x.feed(string)
    return x

def dir_to_str(directory: str) -> [str]:
    strings = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            file = codecs.open(f, "r", "utf-8")
            strings.append(file.read())
    return strings
