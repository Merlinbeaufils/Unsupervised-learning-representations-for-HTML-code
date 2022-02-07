import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloading2 import *


torch.manual_seed(1)
def bow_model(args):
    vocab_map = args.tags
    embeds = nn.Embedding(len(vocab_map), 600)  # 36 tags in vocab, 600 dimensional embeddings
    # wtf why only 36
    lookup_tensor = torch.tensor([vocab_map["html"]], dtype=torch.long)
    html_embed = embeds(lookup_tensor)
    print(html_embed)
    return html_embed


