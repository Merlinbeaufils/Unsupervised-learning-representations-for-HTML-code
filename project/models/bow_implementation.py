import torch
import torch.nn as nn

torch.manual_seed(1)
def bow_model(args):
    vocab_map = args.tags
    embeds = nn.Embedding(len(vocab_map), 600)  # 36 tags in vocabularies, 600 dimensional embeddings
    # wtf why only 36
    lookup_tensor = torch.tensor([vocab_map["html"]], dtype=torch.long)
    html_embed = embeds(lookup_tensor)
    print(html_embed)
    return html_embed


