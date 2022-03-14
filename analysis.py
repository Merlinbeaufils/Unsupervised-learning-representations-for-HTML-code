from __future__ import annotations
from project.frequency import *

text_file_loc = 'data/common_sites/text_files/'
void_tags = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
             "source", "track", "wbr"]


def data_analysis(skip=False):
    if not skip:
        strings = dir_to_str('data/common_sites')
        with open(text_file_loc + 'new_data.txt', 'w', errors='ignore') as file:
            parser = DataParser(file)
            [parser.feed(string) for string in strings]

    with open(text_file_loc + 'new_data.txt', 'r', errors='ignore') as file:
        data_set = file.read().split("<START> ")
        data_dict = {}
        for data_tuple in data_set:
            split = data_tuple.split(":", 1)
            if split[0]:
                tag, data = split
                if tag in data_dict:
                    if data in data_dict[tag]:
                        data_dict[tag][data] += 1
                    else:
                        data_dict[tag][data] = 1
                else:
                    data_dict[tag] = {data: 1}
    print('done')


class DataParser(HTMLParser):
    def __init__(self, data_file, data_head_file):
        super().__init__()
        self.file = data_file
        self.head_file = data_head_file
        self.tag_stack = ["error"]
        self.void_tag = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.void_tag:
            self.tag_stack.pop()
        self.tag_stack.append(tag)
        if tag in void_tags:
            self.void_tag = 1

    def handle_endtag(self, tag: str) -> None:
        self.tag_stack.pop()

    def handle_data(self, data: str) -> None:
        tag = self.tag_stack[-1]
        if data.strip() and tag:
            self.file.write(tag + ':' + data + "<START> ")


data_analysis()


"""
Comments from analysis of data in common_sites folder:

There are 27 tags which give a data value
    script, 139: Seems completely unuseable and useless. 139 unique scripts.
    title, 10: One title per file. Quite relevant to site. Legible, could go into text_model.
    div, 195: Some repeat. Legible, could go into text model.

If we trim the dict down to only data with more than one occurence, we are left with 210 data occurences. 
Which can be trimmed down a little farther if we get rid of unuseable values.


"""
