import parsing

def main(directory: str):
    directory = directory
    strings = parsing.dir_to_str(directory)
    trees = map(lambda x: parsing.str_to_tree(x), strings)
    return trees

if __name__ == "__main__":
    main('./html_files')