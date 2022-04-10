# Thesis
### Learning vector representations of websites
Dependencies: pandas, pytorch_lightning, sklearn, torch, seaborn
## Structure of repository:
- __data__ - contains different data repos. The final one is the final_feather data folder.
- __project__ - folder contains coding modules, different model implementations and running scripts
- __run.py__ - trains models with specific configurations specified in a script
- __res_and_ckpts__ - contains logs and ckpt files for 
tensorboard and loading models

## Scripts (for running immediately)

- bow_basic_run.sh trains a bow model
- lstm_basic_run.sh trains a lstm model
- transformer_basic_run.sh trains a transformer model

You will probably need to configure a script adapted to your machine and goal.

### Description
<!-- I will investigate techniques to learn representations of websites. In particular, I will use self-supervised learning to train models using NLP algorithms (like masked language modeling) to learn vector
representations. These representations should hold sufficient information to use them for several
downstream tasks, including e.g., analytics and prediction problems.
The first research step will consist of defining the format of the data to be input into the model, to define
datastructures, and loaders. Treating HTML code like a tree, current literature suggests tokenizing nodes in a dictionary-like structure.
The second step is training deep representation models to encode these trees. To do so, I will use self-supervised
learning techniques such as masking to train different target and context encoders, eventually giving us a context encoder
capable of outputting vector representations of entire websites.
In parallel, I will create different metrics to measure the expressive power of these representations. These will include
mask-prediction-accuracy (or similar ranking metrics like recall@3 or mean reciprocal rank) by checking the “closest”
websites (or parts thereof) to the vector representation output by the encoder.
Generating and evaluating this complete and informative website space will consist of the bulk of the thesis.
Finally, I will explore downstream uses of the space. While these are infinite, two uses I am particularly interested in
exploring are a next webpage predictor given a current browsing sequence and a click predictor evaluating the probability
of a user clicking on a website if shown after a search, similar to a ranking algorithm. -->

I have used self-supervised contrastive learning to train vector representations of html files in a masked language modeling framework. We first turn a html byte string into a HtmlNode class in the project.parsing module. We then tokenize this tree into a 2-dimensional tensor in the project.tree-tokenizer module. We define our pytorch dataset class to handle a list of trees and build samples given a configuration in the project.dataloading module. Finally we use a pipeline of different types of models easily implemented by the torch and pytorch lightning modules inside the project.models folder.

When running the run.py module, tree-size, batch-size and dozens more training specs can be specified.

We then implement the finetuning task of guessing the top-level-domain of a website through a classifier. 


## Data and data formatting
Download the final_pretrain_data. as a pandas file set it in the ./data/final_feather directory. Running the code will generate: vocabs, frequency_dict, dataloaders and text_files folders containing different steps of the data processing for future user analysis.


### Data representation:
1) Websites are turned from html strings into trees. Each 
node in the tree represents an element and contains its depth 
in the tree, its tag, its attributes as a list of (key, value) 
tuples, and its data. It also keeps track of its father 
node and list of children nodes.

2) We then deterministically turn the tree into a list using a 
post-order traversal. Using the depth value of each node, the original
tree can easily be recovered.

### Tokenizing the trees into samples
1) We combine the tags, data and the keys, and values from attributes
as a total vocabulary.

2) We tokenize the trees as a list of tokenized nodes determined 
by its post-order traversal.

3) Representing each node as a list
[depth, tag, key, value, key, value,..., key, value, data] we tokenize
it using the vocabulary. Adding len(vocabulary) to the depth value.

4) I am still in the process of effectively representing the data.

## Modeling approach

We use a masked language modeling framework.


tokenized_contexts are of shape (batch_size, tree_size, node_size)

tokenized_labels can be of shape (batch_size, node_size) or the same shape
depending on whether it is just a node or subtree.

    context_reps = context_model(tokenized_contexts)    #(batch_size, embedding_dim) 
    label_reps   = label_model(tokenized_labels)      #(batch_size, embedding_dim)
    scores       = node_reps @ tree_reps.T           #(batch_size, batch_size)
    labels       = [0 ... batch_size]                #(batch_size)
    loss         = cross_entropy_loss(scores, labels)#(0)
    

### Models configurations
Currently three functioning configurations: bow, lstm, transformer

#### BOW configuration ~ __baseline to compare with__:
- context_model and label_model are the same.
- simple embedding layer flattening the 2-d tree arrays and summing
all vector embeddings. Ignoring padding.

#### LSTM configuration ~ __more successful attempt__:
- lstm over node representations with a submodel creating nodes
- currently treats nodes as a bow as there is no sequencial logic
to their tokenized representations.

#### Transformer ~ __final endeavor__:
- implements positional embeddings by flattening the 2-d 
trees and using end of node and start of node tags.
- Does not use above framework.


## Improvement flow:

#### Basic bow:
- Baseline idea. 
- no use of data yet.
- poor accuracy.

#### tree_reduction
- randomnly drop subtrees until tree is of usable length.
- drop all subtrees of too high depth.
- slight improvement.

#### lstm
- learn sequential aspect of trees
- larger improvement. 

#### representations
- embedding_dim
- nodes or full subtrees
- random node masking or full subtrees
- improvement?

#### 1d-cnn, deepwalk, transformer
- significant improvements

#### include data
- after good way of including data hopefully good accurary.


## TODO
- Document code well
- Get Data and clean new data
- Include node data attribute
- reduction techniques
- evaluation task
- Send first real experiment.

# ignone rest of readme

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


### dataloading 2 versus dataloading 
Uses a sparsing function to reduce tree size

### frequency 2 versus frequency
Gets rid of attributes
implements use of Namespace

### for masked language modeling
https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
https://towardsdatascience.com/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598
https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/model.py
https://huggingface.co/transformers/v3.3.1/_modules/transformers/modeling_auto.html
https://cloudacademy.com/course/convolutional-neural-networks/images-as-tensors/
