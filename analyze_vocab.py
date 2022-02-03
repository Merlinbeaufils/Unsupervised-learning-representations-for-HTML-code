from frequency import *
import seaborn as sns


def trim_dict(dictionary: Dict[str, int], floor: int) -> None:
    print(len(dictionary))
    delete = []
    for x in dictionary:
        if dictionary[x] < floor:
            delete.append(x)
    for x in delete:
        del dictionary[x]
    print(len(dictionary))


def analyze_results(file_in: str, file_out: str, pikl=0, floor=0, scale='linear') -> None:
    if pikl == 1:
        with open(file_in[:-4] + "_pickled", 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = word_count(file_in, file_in[:-4] + '_pickled')

    trim_dict(data, floor)
    plot = sns.barplot(x=list(data.keys()), y=list(data.values()))
    plot.set_yscale(scale)
    fig = plot.get_figure()
    fig.savefig(file_out)


# analyze_results(path + 'tag_file.txt', path + 'plots/tag_count.png', pikl=1, floor=2, scale='log')
# analyze_results(path + 'attrs_file.txt', path + 'plots/attrs_count.png')
# analyze_results(path + 'data_file.txt', path + 'plots/data_count.png')
# dict = word_count(path + 'attrs_file.txt', path + 'attrs_file_pickled')


def rebuild():
    path = './analysis/'
    build_files('./common_sites', './analysis')
    word_count(path + 'tag_file.txt', path + 'tag_file_pickled')
    word_count(path + 'attrs_file.txt', path + 'attrs_file_pickled')


def quick_analysis():
    path = './analysis/'
    analyze_results(path + 'tag_file.txt', path + 'plots/' + 'tag_count.png', pikl=1, floor=50, scale='log')
    analyze_results(path + 'attrs_file.txt', path + 'plots/' + 'attrs_count.png', pikl=1, floor=100, scale='linear')


def build_tag_vocabulary(directory):
    build_files(directory, './analysis/tag_file.txt')
    word_count('./analysis/tag_file.txt', './analysis/tag_pickled')
    vocab = []
    with open('./analysis/tag_pickled', 'rb') as handle:
        data = pickle.load(handle)
        for tag in data:
            if data[tag] > sum(data)/100:
                vocab.append(tag)
        with open('./vocabulary/tag_vocab', 'wb') as new_handle:
            pickle.dump(vocab, new_handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
def build_attrs_maps(directory: str, attr_floor=50, key_floor=100,value_floor=100) -> None:
    build_files(directory, './analysis/attrs_file.txt')
    word_count('./analysis/attrs_file.txt', './analysis/attrs_pickled')
    word_count('./analysis/key_file.txt', './analysis/key_pickled')
    word_count('./analysis/value_file.txt', './analysis/value_pickled')

    attr_vocab, key_vocab, value_vocab = Vocabulary(0), Vocabulary(0), Vocabulary(0)
    with open('./analysis/attr_pickled', 'rb') as handle_attr, \
            open('./analysis/key_pickled', 'rb') as handle_key, \
            open('./analysis/value_pickled', 'rb') as handle_value:
        data_attr = pickle.load(handle_attr)
        data_key = pickle.load(handle_key)
        data_value = pickle.load(handle_value)
        temp = []
        for attr in data_attr:
            if data_attr[attr] > attr_floor:
                attr_map[attr] = len(attr_map)
            else:
                temp.append(attr)
'''
