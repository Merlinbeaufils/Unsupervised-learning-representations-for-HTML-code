import parsing


def main(directory: str):
    strings = parsing.dir_to_str(directory)
    trees = [parsing.parse_string(x, parsing.MyHTMLParser()).tree for x in strings]

    return trees


if __name__ == "__main__":
    main('./html_files')
