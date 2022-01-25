# Thesis
Bachelor's thesis


### Description
I will investigate techniques to learn representations of websites. In particular, I will use selfsupervised learning to train models using NLP algorithms (like masked language modeling) to learn vector
representations. These representations should hold sufficient information to use them for several
downstream tasks, including e.g., analytics and prediction problems.
The first research step will consist of defining the format of the data to be input into the model, to define
datastructures, and loaders. Treating HTML code like a tree, current literature suggests tokenizing nodes in a dictionarylike structure.
The second step is training deep representation models to encode these trees. To do so, I will use self-supervised
learning techniques such as masking to train different target and context encoders, eventually giving us a context encoder
capable of outputting vector representations of entire websites.
In parallel, I will create different metrics to measure the expressive power of these representations. These will include
mask-prediction-accuracy (or similar ranking metrics like recall@3 or mean reciprocal rank) by checking the “closest”
websites (or parts thereof) to the vector representation output by the encoder.
Generating and evaluating this complete and informative website space will consist of the bulk of the thesis.
Finally, I will explore downstream uses of the space. While these are infinite, two uses I am particularly interested in
exploring are a next webpage predictor given a current browsing sequence and a click predictor evaluating the probability
of a user clicking on a website if shown after a search, similar to a ranking algorithm.

### Data
repository html_files of html files from somewhere
also common_files the 10 most used websites


## Analysis of tags, attributes and content
#### Build files and word count
Use frequency.buildfiles to build the text files of tags, attributes and data.

You can then make a word count dictionary of each of these using
word_count('tag_file.txt','tag_file_pickled') for example and it will pickle
the dictionary into the file so you can then delete the word files and keep only the counts

#### Analyze
use frequency.analyze to analyze the results. if the pickled file is already 
built, specify pikl=1. Also specify a frequency floor to be sisplayed and the scale.



### Traversal
used this algorithm to traverse tree bottom up
https://www.geeksforgeeks.org/bottom-up-traversal-of-a-trie/